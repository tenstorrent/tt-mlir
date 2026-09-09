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
    auto firstGridShape = inputLayouts[0].getGridShape();
    for (size_t i = 1; i < inputLayouts.size(); ++i) {
      if (inputLayouts[i].getGridShape() != firstGridShape) {
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
    return hint.outputLayout.getGridShape() == inputLayouts[0].getGridShape();
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
// RepeatRuleBook
//===----------------------------------------------------------------------===//

bool RepeatRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  // ttnn.repeat has a single tensor operand.
  if (inputLayouts.empty()) {
    return true;
  }

  auto inputMem = inputLayouts[0].getMemLayout();
  bool inputSharded = inputMem && isShardedMemoryLayout(inputMem.getValue());

  // Interleaved input: no grid constraint (output stays interleaved).
  if (!inputSharded) {
    return true;
  }

  // Sharded input: a NULL/gridless output config lets tt-metal re-derive the
  // output shard spec onto the full compute grid, so the reported grid no
  // longer matches the physically-allocated (trimmed) grid and consumers read
  // the wrong cores (tt-xla#5605). Require an explicit sharded output on the
  // same memory layout type and the same grid as the input.
  if (!hint.outputLayout) {
    return false;
  }
  auto hintMem = hint.outputLayout.getMemLayout();
  bool hintSharded = hintMem && isShardedMemoryLayout(hintMem.getValue());
  if (!hintSharded || hintMem != inputMem) {
    return false;
  }
  return hint.outputLayout.getGridShape() == inputLayouts[0].getGridShape();
}

OutputHints RepeatRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // Promote sharded configs to primary hints so a grid-matching sharded
  // candidate exists for a sharded input. Relying on the NULL hint alone lets
  // tt-metal re-grid the output onto the full compute grid (the tt-xla#5605
  // regression); isValidOutputHintForInputs then keeps only the config whose
  // grid equals the input's. The NULL hint remains for the interleaved-input
  // case (backend picks an interleaved output).
  OutputHints result;
  result.hints.push_back(OpConfig(TTNNLayoutAttr()));
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
bool canReshapeBeView(Operation *op) {
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

//===----------------------------------------------------------------------===//
// View-op detection (input-aliasing / zero-copy ops)
//
// These predicates mirror tt-metal's decision to return the input tensor (or a
// view built at the input's address) instead of allocating a fresh output.
// The L1 spill pass uses `isAliasingViewOp` to alias such an op onto its
// source's L1 slot (see MockAllocatorL1Tracker / addTensorAtAddress) rather
// than tracking a fresh allocation. Under the stateful op-model query, an
// aliasing op's output is reported at the weightless input's address 0, which
// is not a re-installable allocation -- so mis-tracking it crashes
// `with_allocations` (issue
// https://github.com/tenstorrent/tt-mlir/issues/9054). Each predicate is
// guarded by a tripwire unit test asserting that an op it classifies as a view
// really does return address 0 from the mock op-model; if tt-metal drifts, the
// tripwire fails and the predicate must be revisited.
//===----------------------------------------------------------------------===//

/// Check if a `ttnn.pad` returns its input as a view (no new allocation).
/// Mirrors ttnn::pad: all-zero padding (pad.cpp:451), on-device same shape
/// (pad.cpp:128), and the TILE fast-path where the padding stays within the
/// existing tile padding so the tiled buffer is unchanged (pad.cpp:383).
bool canPadBeView(Operation *op) {
  auto pad = mlir::dyn_cast<PadOp>(op);
  if (!pad) {
    return false;
  }
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType) {
    return false;
  }
  llvm::ArrayRef<int32_t> padding = pad.getPadding(); // [lo0,hi0,lo1,hi1,...]

  // (A) All-zero padding is a strict no-op regardless of layout/shape.
  if (llvm::all_of(padding, [](int32_t p) { return p == 0; })) {
    return true;
  }

  llvm::ArrayRef<int64_t> inShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outShape = outputType.getShape();
  if (inShape.size() != outShape.size() || inShape.empty()) {
    return false;
  }

  // (B) Same logical shape -> ttnn::pad returns the input.
  if (inShape == outShape) {
    return true;
  }

  // (C) TILE fast-path: only reachable for tiled layout, with zero front
  // (before) padding, when the change is confined within the tile padding so
  // the tiled (padded) buffer is byte-identical -- then ttnn::pad returns a
  // view at the input's address. tt-metal's condition (pad.cpp:383):
  //   !pad_upper_dims && padded[-1] unchanged && padded[-2] unchanged.
  auto inputLayout = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  if (!inputLayout || !inputLayout.isTiled()) {
    return false;
  }
  // Front (low) padding must be zero on every dim (tt-metal TT_FATALs
  // otherwise; certainly not a view).
  for (size_t d = 0; d < padding.size() / 2; ++d) {
    if (padding[2 * d] != 0) {
      return false;
    }
  }
  // Upper dims (everything but the last two) must be unchanged.
  for (size_t i = 0; i + 2 < inShape.size(); ++i) {
    if (inShape[i] != outShape[i]) {
      return false;
    }
  }
  auto roundUpTo = [](int64_t x, int64_t t) {
    return llvm::divideCeil(x, t) * t;
  };
  int64_t inLast = inShape.back(), outLast = outShape.back();
  int64_t inSecondLast = inShape[inShape.size() - 2];
  int64_t outSecondLast = outShape[outShape.size() - 2];
  return roundUpTo(inLast, TILE_WIDTH) == roundUpTo(outLast, TILE_WIDTH) &&
         roundUpTo(inSecondLast, TILE_HEIGHT) ==
             roundUpTo(outSecondLast, TILE_HEIGHT);
}

/// Check if a `ttnn.repeat` returns its input as a view: an all-ones
/// repetition is a strict no-op (repeat.cpp:196 `return input_tensor`).
bool canRepeatBeView(Operation *op) {
  auto repeat = mlir::dyn_cast<RepeatOp>(op);
  if (!repeat) {
    return false;
  }
  llvm::ArrayRef<int64_t> repeatDims = repeat.getRepeatDims().getShape();
  return llvm::all_of(repeatDims, [](int64_t r) { return r == 1; });
}

/// Check if a `ttnn.permute` is a no-op view. Mirrors tt-metal's
/// `is_permute_nop` (permute.cpp:158): rank<=1, identity permutation, or a
/// permutation that only moves size-1 dims (layout-aware: TILE fixes the last
/// two dims, ROW_MAJOR the last). Permute-nop returns
/// `to_memory_config(input, requested)`, so it aliases the input only when the
/// requested (output) memory config equals the input's -- hence the extra
/// buffer-type + memory-layout equality check.
bool canPermuteBeView(Operation *op) {
  auto permute = mlir::dyn_cast<PermuteOp>(op);
  if (!permute) {
    return false;
  }
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType) {
    return false;
  }
  llvm::ArrayRef<int64_t> dims = permute.getPermutation();
  llvm::ArrayRef<int64_t> shape = inputType.getShape();
  const int64_t rank = static_cast<int64_t>(shape.size());

  bool isNop = false;
  if (rank <= 1) {
    isNop = true;
  } else {
    // Identity permutation.
    bool identity = true;
    for (int64_t i = 0; i < rank; ++i) {
      if (dims[i] != i) {
        identity = false;
        break;
      }
    }
    auto inputLayout = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
    bool tiled = inputLayout && inputLayout.isTiled();
    // TILE: last two dims must stay; ROW_MAJOR: last dim must stay.
    bool lastDimsFixed =
        tiled ? (dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2)
              : (dims[rank - 1] == rank - 1);
    // Only size-1 dims moved (and shape therefore unchanged).
    bool onlySize1Moved = true;
    for (int64_t i = 0; i < rank; ++i) {
      if (i != dims[i] && shape[i] > 1) {
        onlySize1Moved = false;
        break;
      }
    }
    isNop = identity || (lastDimsFixed && onlySize1Moved);
  }
  if (!isNop) {
    return false;
  }

  // Aliases the input only if the output memory config matches the input's.
  auto inLayout = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  auto outLayout = mlir::dyn_cast<TTNNLayoutAttr>(outputType.getEncoding());
  if (!inLayout || !outLayout) {
    return false;
  }
  return inLayout.getBufferType() == outLayout.getBufferType() &&
         inLayout.getMemLayoutOpt() == outLayout.getMemLayoutOpt();
}

bool isAliasingViewOp(Operation *op) {
  return canReshapeBeView(op) || canPadBeView(op) || canRepeatBeView(op) ||
         canPermuteBeView(op);
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
