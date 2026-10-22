// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/DataMovementRules.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>

namespace mlir::tt::ttnn {

// A/B switch for the optimizer rules in this file: TTMLIR_OPT_DISABLE is a
// comma-separated list of rule names that are turned off (release-build
// friendly; used to attribute measured changes to one rule at a time).
static bool optRuleDisabled(llvm::StringRef name) {
  static const char *env = std::getenv("TTMLIR_OPT_DISABLE");
  if (!env) {
    return false;
  }
  llvm::SmallVector<llvm::StringRef> parts;
  llvm::StringRef(env).split(parts, ',', -1, false);
  return llvm::is_contained(parts, name);
}

//===----------------------------------------------------------------------===//
// ConcatRuleBook
//===----------------------------------------------------------------------===//

// Kill switch (comma list in TTMLIR_OPT_DISABLE) for the L1-interleaved
// concat output hints.
static bool concatL1InterleavedDisabled() {
  static const bool disabled = [] {
    const char *env = std::getenv("TTMLIR_OPT_DISABLE");
    return env && llvm::StringRef(env).contains("concat_l1_interleaved");
  }();
  return disabled;
}

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

  // Interleaved inputs can only produce an interleaved output, and the NULL
  // hint puts it in DRAM; offer the L1-interleaved configs too so a concat of
  // small L1 tensors stays in L1 instead of a DRAM round trip in front of its
  // consumer. TTMLIR_OPT_DISABLE=concat_l1_interleaved keeps the DRAM-only
  // behaviour.
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!concatL1InterleavedDisabled() && outputType) {
    bool haveL1Interleaved = false;
    for (const auto &cfg : legalConfigs) {
      if (!cfg.outputLayout || !cfg.outputLayout.hasL1BufferType()) {
        continue;
      }
      auto memLayout = cfg.outputLayout.getMemLayout();
      if (memLayout && !isShardedMemoryLayout(memLayout.getValue())) {
        result.hints.push_back(cfg);
        haveL1Interleaved = true;
      }
    }
    // The legal configs only carry sharded layouts for this op; build the
    // L1-interleaved layout from the result's own layout.
    auto resultLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(outputType.getEncoding());
    if (!haveL1Interleaved && resultLayout) {
      ttcore::DeviceAttr device = ttcore::lookupDevice(op);
      TTNNLayoutAttr l1Interleaved =
          TTNNLayoutAttr::Builder(resultLayout, outputType.getShape())
              .setBufferType(BufferType::L1)
              .setMemoryLayout(TensorMemoryLayout::Interleaved)
              .setGridShape(device.getWorkerGrid().getShape())
              .build();
      result.hints.push_back(OpConfig(l1Interleaved));
    }
  }

  // Reject sharded output hints when the concat output tensor has non-tile-
  // aligned inner dimensions. tt-metal hangs during UntilizeWithUnpadding when
  // a height-sharded tensor has non-tile-aligned penultimate or last dim
  // (e.g. 32x32x17x16).
  // https://github.com/tenstorrent/tt-metal/issues/41504
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

bool isLastDimOnlySlice(Operation *op) {
  auto slice = mlir::dyn_cast_or_null<SliceStaticOp>(op);
  if (!slice || optRuleDisabled("slice_sharded")) {
    return false;
  }
  auto inputType = mlir::dyn_cast<RankedTensorType>(slice.getInput().getType());
  if (!inputType || inputType.getRank() < 2) {
    return false;
  }
  auto asInt = [](Attribute a) { return mlir::cast<IntegerAttr>(a).getInt(); };
  ArrayAttr begins = slice.getBegins(), ends = slice.getEnds(),
            step = slice.getStep();
  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(begins.size()) != rank ||
      static_cast<int64_t>(ends.size()) != rank ||
      static_cast<int64_t>(step.size()) != rank) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (asInt(step[d]) != 1) {
      return false;
    }
    if (d == rank - 1) {
      continue;
    }
    if (asInt(begins[d]) != 0 || asInt(ends[d]) != inputType.getDimSize(d)) {
      return false;
    }
  }
  return true;
}

// A height-sharded L1 tile tensor: every core owns whole rows, so a slice
// of the last dim is shard-local (same rows, fewer columns per core).
static bool isHeightShardedL1Tile(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayout();
  return ml && ml.getValue() == TensorMemoryLayout::HeightSharded &&
         layout.hasL1BufferType() && layout.isTiled();
}

LayoutFilterFn
SliceRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  // Slice: sharded inputs produce incorrect results in tt-metal.
  // https://github.com/tenstorrent/tt-metal/issues/39074
  return layout_filter_utils::rejectAllSharded;
}

LayoutFilterFn SliceRuleBook::getInputLayoutFilter(Operation *op,
                                                   unsigned operandIdx) const {
  // A last-dim-only slice is a per-core operation on a height-sharded tile
  // tensor (the rows a core owns do not change), which tt-metal's slice
  // handles when the output keeps the input's grid and shard height. Accept
  // that input layout; everything else keeps the interleaved-only rule.
  if (isLastDimOnlySlice(op)) {
    return [](TTNNLayoutAttr layout) {
      return isHeightShardedL1Tile(layout) ||
             layout_filter_utils::rejectAllSharded(layout);
    };
  }
  return getInputLayoutFilter(operandIdx);
}

bool SliceRuleBook::shouldExploreReshards() const { return false; }

bool SliceRuleBook::generatesTiledInputSiblings(Operation *op,
                                                unsigned operandIdx) const {
  // A last-dim-only slice of a row-major height-sharded producer (a pool
  // output, for instance) can read the producer tilized on its own shards
  // instead of through a DRAM round trip.
  return operandIdx == 0 && isLastDimOnlySlice(op) &&
         !optRuleDisabled("slice_tiled_sibling");
}

OutputHints
SliceRuleBook::getOutputHints(Operation * /*op*/,
                              const std::vector<OpConfig> &legalConfigs) const {
  // SliceStaticOp, SliceDynamicOp: disabled due to
  // https://github.com/tenstorrent/tt-metal/issues/38016
  // TODO(rpavlovicTT): re-enable slice ops.
  return layout_filter_utils::nonShardedOutputHints(legalConfigs);
}

std::vector<OpConfig> SliceRuleBook::inputDerivedOutputHints(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  // Height-sharded input of a last-dim-only slice: propose the input's own
  // sharding (grid and core range set unchanged, shard height unchanged) for
  // the narrower output. The legal-layout enumeration only knows rectangular
  // grids, so a 63-core producer layout can only be matched this way.
  if (inputLayouts.empty() || !isLastDimOnlySlice(op) ||
      !isHeightShardedL1Tile(inputLayouts[0])) {
    return {};
  }
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!outputType) {
    return {};
  }
  TTNNLayoutAttr derived =
      TTNNLayoutAttr::Builder(inputLayouts[0], outputType.getShape()).build();
  return {OpConfig(derived)};
}

bool SliceRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.empty() || !hint.outputLayout) {
    return true;
  }
  auto inMl = inputLayouts[0].getMemLayout();
  bool inputSharded = inMl && isShardedMemoryLayout(inMl.getValue());
  auto hintMl = hint.outputLayout.getMemLayout();
  bool hintSharded = hintMl && isShardedMemoryLayout(hintMl.getValue());
  if (!inputSharded) {
    // Interleaved input: the interleaved-only rule stands.
    return !hintSharded;
  }
  // Sharded input (only reachable for a last-dim-only slice): the output must
  // keep the input's grid, i.e. be the derived hint.
  return hintSharded && hintMl == inMl &&
         hint.outputLayout.getGridShape() == inputLayouts[0].getGridShape();
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

bool reshapePrefersRowMajorInput(Operation *op) {
  auto reshape = mlir::dyn_cast_or_null<ReshapeOp>(op);
  if (!reshape || optRuleDisabled("reshape_rm_tail")) {
    return false;
  }
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(reshape.getInput().getType());
  auto outputType = mlir::dyn_cast<RankedTensorType>(reshape.getType());
  if (!inputType || !outputType) {
    return false;
  }
  auto layout = mlir::dyn_cast<TTNNLayoutAttr>(inputType.getEncoding());
  if (!layout || !layout.isTiled() || canReshapeBeView(op)) {
    return false;
  }
  // In ROW_MAJOR the reshape is a view iff the last dim is unchanged.
  auto inShape = inputType.getShape(), outShape = outputType.getShape();
  if (inShape.empty() || outShape.empty() ||
      inShape.back() != outShape.back()) {
    return false;
  }
  // Only when every user reads row-major (a permute, or a grid_sample whose
  // kernel is row-major only): a tiled consumer would need the tensor
  // re-tilized, which costs more than the copy saved here.
  return !reshape->use_empty() &&
         llvm::all_of(reshape->getUsers(), [](Operation *user) {
           return isa<PermuteOp>(user) ||
                  (isa<GridSampleOp>(user) &&
                   !optRuleDisabled("reshape_rm_gridsample"));
         });
}

bool permuteIsConvFront(Operation *op) {
  auto permute = mlir::dyn_cast_or_null<PermuteOp>(op);
  if (!permute || optRuleDisabled("permute_rm_front")) {
    return false;
  }
  auto perm = permute.getPermutation();
  if (perm.size() != 4 || perm[0] != 0 || perm[1] != 2 || perm[2] != 3 ||
      perm[3] != 1) {
    return false;
  }
  return !permute->use_empty() &&
         llvm::all_of(permute->getUsers(), [](Operation *user) {
           return isa<ReshapeOp, Conv2dOp>(user);
         });
}

bool permuteIsReturnTail(Operation *op) {
  auto permute = mlir::dyn_cast_or_null<PermuteOp>(op);
  if (!permute || optRuleDisabled("permute_rm_tail")) {
    return false;
  }
  auto perm = permute.getPermutation();
  if (perm.size() != 4 || perm[0] != 0 || perm[1] != 3 || perm[2] != 1 ||
      perm[3] != 2) {
    return false;
  }
  // The row-major input is untilized on its producer's shards: its sticks
  // (the last dim) must stay aligned, which a tile-width multiple guarantees.
  // A narrower last dim (a 12-channel output, say) is left to the tiled path
  // and the output untilize with unpadding.
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(permute.getInput().getType());
  if (!inputType || inputType.getRank() < 2 ||
      inputType.getShape().back() % TILE_WIDTH != 0) {
    return false;
  }
  return !permute->use_empty() &&
         llvm::all_of(permute->getUsers(), [](Operation *user) {
           return isa<func::ReturnOp>(user);
         });
}

/// Row-major L1 height-sharded layout of `type` on the canonical core
/// placement with the shard height a height-sharded conv2d picks for the
/// tensor: ceil(rows / cores) rounded up to a tile, cores = ceil(rows /
/// shard height). Null when the device is unknown or the tensor is too small.
static TTNNLayoutAttr convHeightShardedRowMajorLayout(Operation *op,
                                                      RankedTensorType type,
                                                      TTNNLayoutAttr like) {
  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr || type.getRank() < 2) {
    return nullptr;
  }
  llvm::ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int64_t workerVolume =
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape());
  if (rows < TILE_HEIGHT || workerVolume < 1) {
    return nullptr;
  }
  int64_t shardH = llvm::divideCeil(llvm::divideCeil(rows, workerVolume),
                                    static_cast<int64_t>(TILE_HEIGHT)) *
                   TILE_HEIGHT;
  int64_t cores = llvm::divideCeil(rows, shardH);
  Type rmElementType = ttnn::utils::getElementType(
      like.getContext(), Layout::RowMajor, like.getDataType());
  TTNNLayoutAttr rm = TTNNLayoutAttr::Builder(like, shape)
                          .setElementType(rmElementType)
                          .setBufferType(BufferType::L1)
                          .setMemoryLayout(TensorMemoryLayout::HeightSharded)
                          .setGridShape({cores, 1})
                          .setCoreRangeSet(CoreRangeSetAttr())
                          .buildWithCanonicalCorePlacement(deviceAttr);
  llvm::SmallVector<int64_t> rmShard = rm.getShardShape();
  if (rmShard.size() != 2) {
    return nullptr;
  }
  if (rmShard[0] == shardH) {
    return rm;
  }
  rmShard[0] = shardH;
  MemRefType memref = ttcore::buildMemRef<BufferType, BufferTypeAttr>(
      rm.getContext(), rmShard, rm.getElementType(), rm.getBufferType());
  return TTNNLayoutAttr::get(rm.getContext(), rm.getLinear(), rm.getGridShape(),
                             memref, rm.getMemLayout(), rm.getTensorMesh(),
                             rm.getIgnorePhysicalLayout(),
                             rm.getCoreRangeSet());
}

std::vector<OpConfig> ReshapeRuleBook::inputDerivedOutputHints(
    Operation *op, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.empty() || !inputLayouts[0] ||
      inputLayouts[0].getLayout() != Layout::RowMajor) {
    return {};
  }
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType || inputType.getRank() == 0 ||
      outputType.getRank() == 0) {
    return {};
  }
  if (isa<ReshapeOp>(op)) {
    // The reshape behind a conv-front permute, fed row-major with the last
    // dim unchanged, is a view: the NULL hint keeps the input's sharding (the
    // static hints are tiled interleaved configs that would copy the tensor
    // out of its shards). Other reshapes keep their own rules.
    if (inputType.getShape().back() == outputType.getShape().back() &&
        permuteIsConvFront(op->getOperand(0).getDefiningOp())) {
      return {OpConfig(TTNNLayoutAttr())};
    }
    return {};
  }
  if (permuteIsReturnTail(op)) {
    // Row-major straight to DRAM: the layout the host reads.
    Type rmElementType = ttnn::utils::getElementType(
        op->getContext(), Layout::RowMajor, inputLayouts[0].getDataType());
    TTNNLayoutAttr dram =
        TTNNLayoutAttr::Builder(inputLayouts[0], outputType.getShape())
            .setElementType(rmElementType)
            .setBufferType(BufferType::DRAM)
            .setMemoryLayout(TensorMemoryLayout::Interleaved)
            .build();
    return {OpConfig(dram)};
  }
  if (!permuteIsConvFront(op)) {
    return {};
  }
  TTNNLayoutAttr derived =
      convHeightShardedRowMajorLayout(op, outputType, inputLayouts[0]);
  if (!derived) {
    return {};
  }
  return {OpConfig(derived)};
}

LayoutScore
ReshapeRuleBook::adjustScore(Operation *op, LayoutScore base,
                             const OpConfig &config,
                             llvm::ArrayRef<TTNNLayoutAttr> /*inputLayouts*/,
                             bool /*requiresReshard*/) const {
  if (permuteIsReturnTail(op) && config.outputLayout &&
      config.outputLayout.isTiled()) {
    base.costClass = 1;
  }
  return base;
}

LayoutFilterFn
ReshapeRuleBook::getInputLayoutFilter(unsigned /*operandIdx*/) const {
  // Reshape/Permute: width-sharded inputs round-trip through interleaved
  // internally in tt-metal; height/block sharding is handled natively.
  // https://github.com/tenstorrent/tt-mlir/issues/7681
  return layout_filter_utils::rejectWidthSharded;
}

LayoutFilterFn
ReshapeRuleBook::getInputLayoutFilter(Operation *op,
                                      unsigned operandIdx) const {
  if (reshapePrefersRowMajorInput(op)) {
    // Keep only the ROW_MAJOR siblings generated for this operand (see
    // generatesRowMajorInputSiblings): the tiled originals would copy.
    return [](TTNNLayoutAttr layout) {
      return layout_filter_utils::requireRowMajor(layout) &&
             layout_filter_utils::rejectWidthSharded(layout);
    };
  }
  return getInputLayoutFilter(operandIdx);
}

bool ReshapeRuleBook::generatesRowMajorInputSiblings(
    Operation *op, unsigned operandIdx) const {
  // The conv-front permute keeps its tiled candidates as well: the row-major
  // ones only win when the sharded row-major output validates.
  return operandIdx == 0 && (reshapePrefersRowMajorInput(op) ||
                             permuteIsConvFront(op) || permuteIsReturnTail(op));
}

bool ReshapeRuleBook::shouldExploreReshards() const { return false; }

OutputHints ReshapeRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {
  if (canReshapeBeView(op) || reshapePrefersRowMajorInput(op)) {
    // View-eligible reshape (or one that becomes a view once its input is the
    // ROW_MAJOR sibling, see reshapePrefersRowMajorInput): use NULL hint only
    // (no fallbacks).
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
