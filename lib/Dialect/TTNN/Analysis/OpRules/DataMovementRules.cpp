// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ConcatRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn
ConcatRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  // Concat sharded inputs: re-enabled after tt-metal hang fix landed.
  // https://github.com/tenstorrent/tt-metal/issues/39419
  // Fix: https://github.com/tenstorrent/tt-metal/pull/39882
  return nullptr;
}

bool ConcatRuleBook::shouldExploreReshards() const { return true; }

bool ConcatRuleBook::isValidInputCombination(
    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.size() < 2) {
    return true;
  }

  // Concat requires all inputs to have the same memory layout type.
  auto firstMem = inputLayouts[0].getMemLayout();
  for (size_t i = 1; i < inputLayouts.size(); ++i) {
    if (inputLayouts[i].getMemLayout() != firstMem) {
      return false;
    }
  }

  // Concat requires all sharded inputs to have the same grid.
  if (inputLayouts[0].hasShardedTensorMemoryLayout()) {
    auto firstGrid = inputLayouts[0].getGrid();
    for (size_t i = 1; i < inputLayouts.size(); ++i) {
      if (inputLayouts[i].getGrid() != firstGrid) {
        return false;
      }
    }
  }

  return true;
}

bool ConcatRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.empty()) {
    return true;
  }

  auto inputMem = inputLayouts[0].getMemLayout();
  bool inputsSharded = inputMem && isShardedMemoryLayout(inputMem.getValue());

  // NULL hint defaults to DRAM in tt-metal; reject when inputs are sharded
  // (concat requires sharded output when inputs are sharded).
  if (!hint.outputLayout) {
    return !inputsSharded;
  }

  auto hintMem = hint.outputLayout.getMemLayout();
  bool hintSharded = hintMem && isShardedMemoryLayout(hintMem.getValue());

  // Sharded inputs require sharded output with matching memory layout type
  // and matching grid.
  if (inputsSharded) {
    if (!hintSharded || hintMem != inputMem) {
      return false;
    }
    auto hintGrid = hint.outputLayout.getGrid();
    auto inputGrid = inputLayouts[0].getGrid();
    if (!hintGrid || !inputGrid) {
      return false;
    }
    return hintGrid == inputGrid;
  }

  // Interleaved inputs: reject sharded output hints. tt-metal concat selects
  // the wrong writer kernel for interleaved-to-sharded, causing a JIT failure.
  // https://github.com/tenstorrent/tt-metal/issues/41469
  if (hintSharded) {
    return false;
  }

  return true;
}

OutputHints ConcatRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {
  // tt-metal concat defaults to DRAM_MEMORY_CONFIG when output memory config is
  // nullopt, so a NULL hint always produces DRAM output. For sharded inputs
  // this immediately fails validation (output must be sharded when inputs are
  // sharded). Promote sharded configs to primary hints to avoid wasting backend
  // calls on the guaranteed-to-fail NULL path.
  OutputHints result;
  result.hints.push_back(OpConfig(TTNNLayoutAttr()));

  // Reject sharded output hints when the concat output tensor has non-tile-
  // aligned inner dimensions. tt-metal hangs during UntilizeWithUnpadding when
  // a height-sharded tensor has non-tile-aligned penultimate or last dim
  // (e.g. 32x32x17x16).
  // https://github.com/tenstorrent/tt-metal/issues/41504
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (outputType) {
    llvm::ArrayRef<int64_t> shape = outputType.getShape();
    int64_t rank = shape.size();
    if (rank >= 2) {
      bool penultimateAligned = (shape[rank - 2] % TILE_HEIGHT) == 0;
      bool lastAligned = (shape[rank - 1] % TILE_WIDTH) == 0;
      if (!penultimateAligned || !lastAligned) {
        return result;
      }
    }
  }

  for (const auto &cfg : legalConfigs) {
    if (!cfg.outputLayout) {
      continue;
    }
    auto memLayout = cfg.outputLayout.getMemLayout();
    if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
      result.hints.push_back(cfg);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// SliceRuleBook
//===----------------------------------------------------------------------===//

LayoutFilterFn
SliceRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  // Slice: sharded inputs produce incorrect results in tt-metal.
  // https://github.com/tenstorrent/tt-metal/issues/39074
  return layout_filter_utils::rejectAllSharded;
}

bool SliceRuleBook::shouldExploreReshards() const { return false; }

OutputHints
SliceRuleBook::getOutputHints(Operation * /*op*/,
                              const std::vector<OpConfig> &legalConfigs) const {
  // SliceStaticOp, SliceDynamicOp: disabled due to
  // https://github.com/tenstorrent/tt-metal/issues/38016
  // TODO(rpavlovicTT): re-enable slice ops.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// ReshapeRuleBook
//===----------------------------------------------------------------------===//

/// Check if a reshape can be optimized to a view (no kernel launch) by
/// tt-metal. Mirrors the `this_is_view` condition in tt-metal's
/// reshape.cpp:378-384. Only applies to ReshapeOp — PermuteOp has its own
/// NOP optimization (is_permute_nop) with different conditions.
/// TODO(#7988): Split PermuteOp into its own PermuteRuleBook to avoid this
/// op-kind check.
static bool canReshapeBeView(Operation *op) {
  if (!mlir::isa<ReshapeOp>(op)) {
    return false;
  }
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType) {
    return false;
  }

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  if (inputShape.empty() || outputShape.empty()) {
    return false;
  }

  // Condition 1: last dimension must be unchanged.
  if (inputShape.back() != outputShape.back()) {
    return false;
  }

  // Condition 2: for tiled layout, second-to-last dim must be unchanged or
  // both second-to-last dims must be tile-aligned (no padding change).
  auto ttnnLayout = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  if (ttnnLayout && ttnnLayout.isTiled()) {
    int64_t inputSecondLast =
        inputShape.size() >= 2 ? inputShape[inputShape.size() - 2] : 1;
    int64_t outputSecondLast =
        outputShape.size() >= 2 ? outputShape[outputShape.size() - 2] : 1;
    if (inputSecondLast != outputSecondLast &&
        !(outputSecondLast % TILE_HEIGHT == 0 &&
          inputSecondLast % TILE_HEIGHT == 0)) {
      return false;
    }
  }

  return true;
}

LayoutFilterFn
ReshapeRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  // Reshape/Permute: width-sharded inputs round-trip through interleaved
  // internally in tt-metal; height/block sharding is handled natively.
  // https://github.com/tenstorrent/tt-mlir/issues/7681
  return layout_filter_utils::rejectWidthSharded;
}

bool ReshapeRuleBook::shouldExploreReshards() const { return false; }

OutputHints ReshapeRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {
  if (canReshapeBeView(op)) {
    // View-eligible reshape: use NULL hint only (no fallbacks).
    // tt-metal's reshape inherits input memory config when output config is
    // nullopt (reshape.cpp:337), which guarantees the memory-type and sharding
    // conditions for the view optimization (reshape.cpp:378-384) are met.
    // By not providing L1 fallback hints, we prevent the greedy optimizer from
    // upgrading DRAM→L1 and breaking the zero-cost view path.
    return layout_filter_utils::nullHintOnly();
  }
  // Non-view reshape: sharded output not beneficial, use non-sharded configs.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// PadRuleBook
//===----------------------------------------------------------------------===//

bool PadRuleBook::shouldExploreReshards() const { return false; }

OutputHints
PadRuleBook::getOutputHints(Operation * /*op*/,
                            const std::vector<OpConfig> &legalConfigs) const {
  // PadOp: ttnn::pad only supports sharded output for HEIGHT_SHARDED +
  // ROW_MAJOR inputs. Its tile-layout path (invoke_tile) crashes when given a
  // sharded output memory config with interleaved input because it
  // unconditionally dereferences input_tensor.shard_spec()->grid, which is
  // nullopt for interleaved tensors. The op model validation doesn't catch
  // this since it only checks sharding constraints when the input is already
  // sharded.
  // https://github.com/tenstorrent/tt-metal/issues/40898
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

//===----------------------------------------------------------------------===//
// MeshPartitionRuleBook
//===----------------------------------------------------------------------===//

bool MeshPartitionRuleBook::shouldExploreReshards() const { return false; }

OutputHints MeshPartitionRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // mesh_partition: feed downstream KV-cache and paged-SDPA-decode consumers
  // with DRAM-interleaved tensors only. Letting the optimizer pick L1 here
  // produces an L1-block-sharded output that the SDPA-decode kernel rejects;
  // when the multi-consumer reshard logic does not insert a DRAM reshard for
  // every consumer, the verifier fires `keyType != valueType`.
  return layout_filter_utils::dramInterleavedOnlyOutputHints(legalConfigs);
}

} // namespace mlir::tt::ttnn
