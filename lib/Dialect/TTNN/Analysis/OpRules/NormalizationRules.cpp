// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/NormalizationRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {

namespace {

/// Input operand's tile-height: the total number of tile rows across all
/// leading (batch/channel) dims times the tile-aligned height of the
/// last-two-dim row. Computed as `NC * Ht`, where `NC` is the product of all
/// dims except the last two and `Ht = ceil(H / TILE_HEIGHT)`. Kernels that
/// parallelize by distributing tile rows across cores are capped by this
/// count regardless of how the output tensor is sharded.
int64_t inputOperandTileHeight(Operation *op, unsigned operandIdx) {
  auto opType =
      mlir::cast<mlir::RankedTensorType>(op->getOperand(operandIdx).getType());
  assert(opType && "expected ranked tensor operand");

  auto shape = opType.getShape();
  constexpr int64_t kTileHeight = 32;
  int64_t ht = shape.size() >= 2
                   ? llvm::divideCeil(shape[shape.size() - 2], kTileHeight)
                   : 1;
  int64_t nc = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    nc *= shape[i];
  }
  return nc * ht;
}

} // namespace

//===----------------------------------------------------------------------===//
// RmsNormRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn
RmsNormRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  // Only operand 0 (activation) gates kernel dispatch. Gamma (operand 1) and
  // beta (operand 2) are small 1D tensors; leave them unconstrained.
  if (operandIdx != 0) {
    return nullptr;
  }

  return [](TTNNLayoutAttr layout) -> bool {
    auto ml = layout.getMemLayout();
    if (!ml) {
      return true; // no mem layout info — accept
    }

    auto value = ml.getValue();

    // Interleaved is always accepted.
    if (!isShardedMemoryLayout(value)) {
      return true;
    }

    // Height-sharded is explicitly rejected by the sharded layernorm kernel
    // (layernorm_device_operation.cpp:151).
    if (value == TensorMemoryLayout::HeightSharded) {
      return false;
    }

    // Block- and width-sharded are accepted, provided the grid is a full
    // rectangular bbox.
    return layout_filter_utils::isFullBboxSharded(layout);
  };
}

LayoutScore RmsNormRuleBook::adjustScore(
    Operation *op, LayoutScore base, const OpConfig &config,
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts, bool requiresReshard) const {
  // Only operand 0 (activation) gates rms_norm kernel dispatch.
  if (inputLayouts.empty() || !inputLayouts[0]) {
    return base;
  }
  TTNNLayoutAttr operand0Layout = inputLayouts[0];
  auto ml = operand0Layout.getMemLayout();

  if (isShardedMemoryLayout(ml.getValue())) {
    // Sharded-input path: the sharded layernorm kernel parallelizes across
    // the input shard grid. The output-derived `coreCount` already typically
    // matches (kernel writes to the same grid), but force it to the input
    // grid's volume for ops where output sharding might differ. Also clear
    // `requiresReshard` — the 3x+ kernel speedup over the interleaved variant
    // dominates the few-µs reshard cost on decode-shape tensors.
    base.coreCount =
        static_cast<int64_t>(operand0Layout.getGrid().getGridVolume());
    base.requiresReshard = false;
  } else {
    // Interleaved-input path: `LayerNormMultiCoreProgramFactory` parallelizes
    // by tile-row count. Override the output-derived coreCount with the
    // input operand's tile height so the comparison reflects kernel reality,
    // not output distribution.
    base.coreCount = inputOperandTileHeight(op, /*operandIdx=*/0);
  }
  return base;
}

bool RmsNormRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.empty() || !inputLayouts[0]) {
    return true;
  }
  TTNNLayoutAttr inputLayout = inputLayouts[0];
  auto inputMem = inputLayout.getMemLayout();
  bool inputSharded = inputMem && isShardedMemoryLayout(inputMem.getValue());

  if (!inputSharded) {
    return true; // interleaved factory handles any output layout
  }

  // NULL hint defers to the backend; tt-metal's sharded layernorm defaults
  // its output mem config to the input mem config, which yields
  // skip_write_back=true. Safe.
  if (!hint.outputLayout) {
    return true;
  }

  auto hintMem = hint.outputLayout.getMemLayout();
  if (!hintMem || !isShardedMemoryLayout(hintMem.getValue())) {
    // Sharded inputs require sharded outputs (validate enforces this).
    return false;
  }
  // Same memory layout type (block/width), same grid, same shard shape →
  // skip_write_back path, no integrated reshard.
  if (hintMem.getValue() != inputMem.getValue()) {
    return false;
  }
  if (inputLayout.getGrid() != hint.outputLayout.getGrid()) {
    return false;
  }
  if (inputLayout.getShardShape() != hint.outputLayout.getShardShape()) {
    return false;
  }
  return true;
}

} // namespace mlir::tt::ttnn
