// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/NormalizationRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

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
    base.coreCount = ttmlir::utils::volume(operand0Layout.getGridShape());
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

//===----------------------------------------------------------------------===//
// DistributedRMSNormRuleBook
//===----------------------------------------------------------------------===//

namespace {

constexpr int64_t kTileWidth = 32;

// The fused kernel runs only on a full-bbox WIDTH-sharded L1 input.
bool isWidthShardedL1FullBbox(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  if (!ml || ml.getValue() != TensorMemoryLayout::WidthSharded) {
    return false;
  }
  if (layout.getBufferType() != BufferType::L1) {
    return false;
  }
  return layout_filter_utils::isFullBboxSharded(layout);
}

// Derive the LayerNormShardedMultiCoreProgramConfig from a width-sharded
// layout, mirroring DistributedRMSNormWidthShardInputRewritePattern: grid = the
// shard core grid's bounding box (full-bbox => rectangular), block_h/block_w
// from the per-core shard shape. Returns null if the layout has no usable shard
// spec.
LayerNormShardedMultiCoreProgramConfigAttr
programConfigFromLayout(mlir::MLIRContext *ctx, TTNNLayoutAttr layout) {
  auto memConfig = MemoryConfigAttr::get(layout);
  std::optional<ShardSpecAttr> shardSpec = memConfig.getShardSpec();
  if (!shardSpec.has_value()) {
    return nullptr;
  }
  std::optional<CoreRangeAttr> bbox =
      shardSpec->getCoreRangeSet().getBoundingBox();
  if (!bbox.has_value()) {
    return nullptr;
  }
  CoreCoordAttr start = bbox->getStartCoord();
  CoreCoordAttr end = bbox->getEndCoord();
  uint64_t gridW = end.getX() - start.getX() + 1;
  uint64_t gridH = end.getY() - start.getY() + 1;

  llvm::SmallVector<int64_t> shardShape = layout.getScalarShardShape();
  if (shardShape.size() < 2) {
    return nullptr;
  }
  int64_t blockH = shardShape[0] / kTileWidth;
  int64_t blockW = shardShape[1] / kTileWidth;

  return LayerNormShardedMultiCoreProgramConfigAttr::get(
      ctx, CoreCoordAttr::get(ctx, gridW, gridH),
      /*subblock_w=*/1, blockH, blockW, /*inplace=*/false);
}

} // namespace

LayoutFilterFn
DistributedRMSNormRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  // operand 0 = input, operand 2 = residual: full-bbox width-sharded L1.
  if (operandIdx == 0 || operandIdx == 2) {
    return [](TTNNLayoutAttr layout) -> bool {
      return isWidthShardedL1FullBbox(layout);
    };
  }
  // operand 1 = weight (gamma): the fused kernel requires ROW_MAJOR.
  if (operandIdx == 1) {
    return layout_filter_utils::requireRowMajor;
  }
  return nullptr;
}

bool DistributedRMSNormRuleBook::generatesRowMajorInputSiblings(
    unsigned operandIdx) const {
  return operandIdx == 1; // weight must be RowMajor
}

OutputHints DistributedRMSNormRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {
  // Output shape == input shape (only stats are all-gathered), so the output
  // must also be full-bbox width-sharded L1. For each such candidate, attach
  // the program config derived from its shard spec.
  std::vector<OpConfig> hints;
  for (const OpConfig &cfg : legalConfigs) {
    if (!cfg.outputLayout || !isWidthShardedL1FullBbox(cfg.outputLayout)) {
      continue;
    }
    LayerNormShardedMultiCoreProgramConfigAttr pc =
        programConfigFromLayout(op->getContext(), cfg.outputLayout);
    if (!pc) {
      continue;
    }
    DistributedRMSNormAttrs attrs;
    attrs.programConfig = pc;
    hints.push_back(OpConfig(cfg.outputLayout,
                             OpConfig::OpSpecificAttrs{std::move(attrs)}));
  }
  return OutputHints{hints, {}};
}

void DistributedRMSNormRuleBook::applyOpSpecificAttrs(
    Operation *op, const BeamCandidate &candidate) const {
  auto normOp = dyn_cast<DistributedRMSNormOp>(op);
  if (!normOp) {
    return;
  }
  if (!std::holds_alternative<DistributedRMSNormAttrs>(
          candidate.configHint.opSpecificAttrs)) {
    return;
  }
  const DistributedRMSNormAttrs &attrs =
      std::get<DistributedRMSNormAttrs>(candidate.configHint.opSpecificAttrs);
  if (!attrs.programConfig.has_value() || !attrs.programConfig.value()) {
    return;
  }
  normOp.setProgramConfigAttr(attrs.programConfig.value());
}

LayoutScore DistributedRMSNormRuleBook::adjustScore(
    Operation *op, LayoutScore base, const OpConfig &config,
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts, bool requiresReshard) const {
  // Same reasoning as RMSNorm: the sharded kernel parallelizes across the input
  // shard grid, and the kernel speedup dominates any input reshard cost.
  if (inputLayouts.empty() || !inputLayouts[0]) {
    return base;
  }
  TTNNLayoutAttr operand0Layout = inputLayouts[0];
  auto ml = operand0Layout.getMemLayout();
  if (ml && isShardedMemoryLayout(ml.getValue())) {
    base.coreCount = ttmlir::utils::volume(operand0Layout.getGridShape());
    base.requiresReshard = false;
  }
  return base;
}

} // namespace mlir::tt::ttnn
