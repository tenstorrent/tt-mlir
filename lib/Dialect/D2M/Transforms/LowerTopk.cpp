// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/TopKShardingStrategy.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERTOPK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileWidth = ttcore::TileType::getDefaultShape()[1];

//===----------------------------------------------------------------------===//
// Layout and region helpers
//===----------------------------------------------------------------------===//

/// Rebuild `layout`'s logical shape from a [grid..., shard...] `deviceShape` of
/// even rank, carrying every other layout field over unchanged.
ttcore::MetalLayoutAttr
rebuildLayoutForDeviceShape(ttcore::MetalLayoutAttr layout,
                            ArrayRef<int64_t> deviceShape) {
  TT_assert(deviceShape.size() % 2 == 0u);
  const std::size_t physicalRank = deviceShape.size() / 2;
  const int64_t tileDim = ttcore::TileType::getDefaultShape()[0];

  llvm::SmallVector<int64_t> logicalShape;
  logicalShape.reserve(physicalRank);
  for (std::size_t i = 0; i < physicalRank; ++i) {
    logicalShape.push_back(deviceShape[i] * deviceShape[i + physicalRank] *
                           tileDim);
  }
  return ttcore::MetalLayoutAttr::get(
      layout.getContext(), logicalShape, layout.getDimAlignments(),
      layout.getCollapsedIntervals(), layout.getMemorySpace(),
      layout.getMemoryLayout());
}

//===----------------------------------------------------------------------===//
// Recovering TTIRToD2M's single-core lowering
//===----------------------------------------------------------------------===//

/// The pieces this pass replaces, all reachable from the leaf `topk_block`.
struct SingleCoreTopK {
  GenericOp leaf;
  /// Pre-layout input, so the pass shards it rather than re-sharding a layout.
  Value logicalInput;
  GenericOp extractValues;
  GenericOp extractIndices;
  /// Follows the index extract when `dim != 0`, else null.
  GenericOp indexCast;
};

/// Walks back from the leaf's input through the `to_layout`s that materialize
/// buffers, the padding-tail `mask`, and the tile-transpose generic dim=1 adds.
Value findLogicalInput(Value value) {
  auto isDevice = [](Value v) {
    auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
    return type &&
           mlir::isa_and_nonnull<ttcore::MetalLayoutAttr>(type.getEncoding());
  };
  while (isDevice(value)) {
    Operation *def = value.getDefiningOp();
    if (auto toLayout = mlir::dyn_cast_if_present<ToLayoutOp>(def)) {
      value = toLayout.getInput();
    } else if (auto mask = mlir::dyn_cast_if_present<MaskOp>(def)) {
      value = mask.getInput();
    } else if (auto generic = mlir::dyn_cast_if_present<GenericOp>(def);
               generic && generic.getInputs().size() == 1) {
      value = generic.getInputs()[0];
    } else {
      return nullptr;
    }
  }
  return value;
}

/// The single generic consuming `value`, looking through the `to_layout` that
/// materializes it. Null when the use pattern is not the one TTIRToD2M emits.
GenericOp findConsumingGeneric(Value value) {
  for (int hop = 0; hop < 2; ++hop) {
    Operation *consumer = nullptr;
    for (Operation *user : value.getUsers()) {
      // A generic reads its operands back through `remote_load`, so each
      // operand has a nested use as well as a direct one, both naming that
      // generic.
      if (GenericOp owner = user->getParentOfType<GenericOp>()) {
        user = owner;
      }
      if (consumer && consumer != user) {
        return nullptr;
      }
      consumer = user;
    }
    if (auto generic = mlir::dyn_cast_if_present<GenericOp>(consumer)) {
      return generic;
    }
    // to_layout is DPS: a use as its output does not continue the chain.
    auto toLayout = mlir::dyn_cast_if_present<ToLayoutOp>(consumer);
    if (!toLayout || toLayout.getInput() != value) {
      return nullptr;
    }
    value = toLayout.getResult(0);
  }
  return nullptr;
}

mlir::FailureOr<SingleCoreTopK> recoverSingleCoreTopK(GenericOp leaf) {
  SingleCoreTopK recovered;
  recovered.leaf = leaf;
  recovered.logicalInput = findLogicalInput(leaf.getInputs()[0]);
  recovered.extractValues = findConsumingGeneric(leaf->getResult(0));
  recovered.extractIndices = findConsumingGeneric(leaf->getResult(1));
  if (!recovered.logicalInput || !recovered.extractValues ||
      !recovered.extractIndices) {
    return mlir::failure();
  }

  // dim != 0 casts the extracted indices to the user's index type in a region
  // of its own; dim == 0 folds that cast into the extract itself.
  GenericOp next = findConsumingGeneric(recovered.extractIndices->getResult(0));
  bool isCast =
      next && next != recovered.extractIndices &&
      next->walk([](TileTypecastOp) { return WalkResult::interrupt(); })
          .wasInterrupted();
  recovered.indexCast = isCast ? next : GenericOp();
  return recovered;
}

/// The logical tensor type a device-layout value stands for.
RankedTensorType getLogicalType(Value deviceValue) {
  auto type = mlir::cast<RankedTensorType>(deviceValue.getType());
  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(type.getEncoding());
  return RankedTensorType::get(
      layout.getLogicalShape(),
      mlir::cast<ttcore::TileType>(type.getElementType()).getElementType());
}

/// Erases `roots` and everything upstream they were the last user of. The chain
/// being replaced is straight-line, so this bottoms out at the logical input.
void eraseDeadChain(RewriterBase &rewriter, ArrayRef<Operation *> roots) {
  llvm::SmallVector<Operation *> worklist(roots);
  llvm::SmallPtrSet<Operation *, 16> erased;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (erased.contains(op) || !op->use_empty()) {
      continue;
    }
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        worklist.push_back(def);
      }
    }
    erased.insert(op);
    rewriter.eraseOp(op);
  }
}

//===----------------------------------------------------------------------===//
// Multi-core emission
//===----------------------------------------------------------------------===//

/// Narrows every core's shard down to its leading `outputReductionTiles`, so
/// the wide partials do not stay live on every core and overflow L1.
Value compactReductionDim(RewriterBase &rewriter, Location loc, Value partial,
                          int64_t outputReductionTiles, int32_t dim) {
  auto wideType = mlir::cast<RankedTensorType>(partial.getType());
  auto tileType = mlir::cast<ttcore::TileType>(wideType.getElementType());
  std::size_t rank = wideType.getShape().size() / 2;
  TopKGeometry geometry = getTopKGeometry(dim, rank);

  llvm::SmallVector<int64_t> narrowShape(wideType.getShape());
  narrowShape[geometry.deviceRedDim] = outputReductionTiles;
  auto narrowEmpty = rewriter.create<EmptyOp>(
      loc, narrowShape, tileType,
      rebuildLayoutForDeviceShape(
          mlir::cast<ttcore::MetalLayoutAttr>(wideType.getEncoding()),
          narrowShape));
  auto narrowType = mlir::cast<RankedTensorType>(narrowEmpty.getType());
  auto generic = rewriter.create<GenericOp>(
      loc, TypeRange{narrowType}, ValueRange{partial},
      ValueRange{narrowEmpty.getResult()}, /*additionalArgs=*/ValueRange(),
      rewriter.getAttr<ttcore::GridAttr>(
          ArrayRef<int64_t>(narrowShape).take_front(rank)),
      /*block_factors=*/rewriter.getI64ArrayAttr({}),
      /*indexing_maps=*/rewriter.getArrayAttr({}),
      /*iterator_types=*/rewriter.getArrayAttr({}),
      rewriter.getArrayAttr(rewriter.getAttr<ThreadAttr>(ThreadType::Unified)),
      /*fabricConnectionConfig=*/nullptr, /*numRegions=*/1);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&generic->getRegion(0));
    auto shardType = [&](RankedTensorType deviceType) {
      return RankedTensorType::get(deviceType.getShape().take_back(rank),
                                   tileType);
    };
    llvm::SmallVector<Value> coreIndices;
    coreIndices.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      coreIndices.push_back(
          rewriter.create<CoreIndexOp>(loc, static_cast<int64_t>(i)));
    }
    auto loadBuf = rewriter.create<tensor::EmptyOp>(
        loc, shardType(wideType).getShape(), tileType);
    Value loaded = rewriter
                       .create<RemoteLoadOp>(loc, shardType(wideType), loadBuf,
                                             partial, coreIndices)
                       .getResult();
    auto outBuf = rewriter.create<tensor::EmptyOp>(
        loc, shardType(narrowType).getShape(), tileType);
    AffineMap shardIdentity = rewriter.getMultiDimIdentityMap(rank);
    Value narrowed =
        rewriter
            .create<LocalCopyOp>(
                loc, loaded, outBuf.getResult(),
                rewriter.getAffineMapArrayAttr({shardIdentity, shardIdentity}))
            .getResult();
    Value stored =
        rewriter
            .create<RemoteStoreOp>(loc, narrowType, narrowEmpty.getResult(),
                                   coreIndices, narrowed)
            .getResult();
    rewriter.create<YieldOp>(loc, ValueRange{stored});
  }
  utils::markPinnedGrid(generic);

  return utils::materializeToLayout(rewriter, loc, generic->getResult(0));
}

/// Lays `rawInput` out banded `gridCols` ways along the reduction dim and/or
/// sliced `ntCores` ways along the non-target dim, then runs one leaf topk per
/// core. Buffers carry logical grids only; GridSelection folds them onto cores.
std::pair<Value, Value> emitShardedLeaves(RewriterBase &rewriter, Location loc,
                                          Value rawInput, int32_t k,
                                          int32_t dim,
                                          const TopKShardingStrategy &strategy,
                                          ttcore::MemorySpace memSpace,
                                          int64_t gridCols, int64_t ntCores) {
  auto inputType = mlir::cast<RankedTensorType>(rawInput.getType());
  llvm::SmallVector<int64_t> shardTiles(inputType.getRank(), 1);
  shardTiles[dim] = strategy.paddedReductionTiles;
  shardTiles[1 - dim] = strategy.paddedNonTargetTiles;
  llvm::SmallVector<int64_t> grid =
      (dim == 1) ? llvm::SmallVector<int64_t>{ntCores, gridCols}
                 : llvm::SmallVector<int64_t>{gridCols, ntCores};
  Value input =
      utils::layoutAndMask(rewriter, loc, rawInput, shardTiles, grid, memSpace);

  auto [vals, idx] = utils::emitLeafTopk(rewriter, loc, input, k, dim,
                                         inputType.getShape()[dim]);
  if (gridCols == 1) {
    return {utils::materializeToLayout(rewriter, loc, vals),
            utils::materializeToLayout(rewriter, loc, idx)};
  }
  return {compactReductionDim(rewriter, loc, vals,
                              strategy.outputReductionTiles, dim),
          compactReductionDim(rewriter, loc, idx, strategy.outputReductionTiles,
                              dim)};
}

/// Collapses `bands` partials, one per core, down to `numGroups`: core g merges
/// the groupTiles bands starting at g * groupTiles, gathered by a composite
/// view that D2MExpandDMAReadCompositeView resolves per tile into a DMA read.
std::pair<Value, Value> emitMergeRound(RewriterBase &rewriter, Location loc,
                                       Value valsIn, Value idxIn, int32_t k,
                                       int32_t dim,
                                       int64_t outputReductionTiles,
                                       int64_t bands, int64_t numGroups) {
  MLIRContext *ctx = rewriter.getContext();
  int64_t groupTiles = bands / numGroups;
  TT_assertv(groupTiles * numGroups == bands,
             "merge round must divide the band count evenly");

  auto valsInType = mlir::cast<RankedTensorType>(valsIn.getType());
  std::size_t rank = valsInType.getShape().size() / 2;
  TopKGeometry geometry = getTopKGeometry(dim, rank);
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<Attribute> iterators(
      rank, ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

  // The input's extent re-split: numGroups cores each holding groupTiles bands'
  // worth of reduction tiles, so the grid x shard product is preserved.
  llvm::SmallVector<int64_t> wideShape(valsInType.getShape());
  wideShape[geometry.deviceGridDim] = numGroups;
  wideShape[geometry.deviceRedDim] = groupTiles * outputReductionTiles;
  auto wideTypeOf = [&](Value in) {
    auto inType = mlir::cast<RankedTensorType>(in.getType());
    return RankedTensorType::get(
        wideShape, inType.getElementType(),
        rebuildLayoutForDeviceShape(
            mlir::cast<ttcore::MetalLayoutAttr>(inType.getEncoding()),
            wideShape));
  };

  // Separate generics: the DMA-expansion pass supports only one composite view
  // per GenericOp (#7600).
  auto gather = [&](Value in) -> Value {
    RankedTensorType wideType = wideTypeOf(in);
    auto composite = rewriter.create<CompositeViewOp>(
        loc, wideType, ValueRange{in}, dim, /*logicalSizes=*/nullptr);
    auto out = rewriter.create<EmptyOp>(
        loc, wideShape, wideType.getElementType(), wideType.getEncoding());
    llvm::SmallVector<Value> ins = {composite.getResult()};
    llvm::SmallVector<Value> outs = {out.getResult()};
    auto generic = rewriter.create<GenericOp>(
        loc, ins, outs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(
            llvm::SmallVector<AffineMap>{identityMap, identityMap}),
        rewriter.getArrayAttr(iterators));
    utils::buildParallelGenericRegion(
        rewriter, loc, generic, ins, outs,
        [](ArrayRef<Value> args) -> llvm::SmallVector<Value> {
          return {args[0]};
        });
    utils::markPinnedGrid(generic);
    return utils::materializeToLayout(rewriter, loc, generic->getResult(0));
  };

  llvm::SmallVector<Value> ins = {gather(valsIn), gather(idxIn)};
  llvm::SmallVector<Value> outs = {
      rewriter
          .create<EmptyOp>(loc, wideShape, valsInType.getElementType(),
                           wideTypeOf(valsIn).getEncoding())
          .getResult(),
      rewriter
          .create<EmptyOp>(
              loc, wideShape,
              mlir::cast<RankedTensorType>(idxIn.getType()).getElementType(),
              wideTypeOf(idxIn).getEncoding())
          .getResult()};
  auto merge = rewriter.create<GenericOp>(
      loc, ins, outs, /*additionalArgs=*/ValueRange(),
      rewriter.getAffineMapArrayAttr(llvm::SmallVector<AffineMap>{
          identityMap, identityMap, identityMap, identityMap}),
      rewriter.getArrayAttr(iterators));
  utils::buildParallelGenericRegion(
      rewriter, loc, merge, ins, outs,
      [&](ArrayRef<Value> args) -> llvm::SmallVector<Value> {
        // No generate_indices: these indices are an earlier stage's results and
        // cannot be recomputed from a coordinate.
        auto block = rewriter.create<TopkBlockOp>(
            loc, args[0], args[1], args[2], args[3], k,
            /*numElements=*/groupTiles * outputReductionTiles * kTileWidth,
            /*stableSort=*/false, dim);
        return {block.getResultValues(), block.getResultIndices()};
      });
  utils::markPinnedGrid(merge);

  return {compactReductionDim(
              rewriter, loc,
              utils::materializeToLayout(rewriter, loc, merge->getResult(0)),
              outputReductionTiles, dim),
          compactReductionDim(
              rewriter, loc,
              utils::materializeToLayout(rewriter, loc, merge->getResult(1)),
              outputReductionTiles, dim)};
}

/// Rebuilds the recovered single-core topk across the grid. `recovered` comes
/// by value because MLIR op accessors are non-const.
void emitMultiCore(RewriterBase &rewriter, SingleCoreTopK recovered, int32_t k,
                   int32_t dim, const TopKShardingStrategy &strategy) {
  Location loc = recovered.leaf->getLoc();
  TopKGeometry geometry = getTopKGeometry(
      dim,
      mlir::cast<RankedTensorType>(recovered.logicalInput.getType()).getRank());

  // Everything the rebuild reads -- the logical input and both extract
  // destinations -- is defined by the values extract.
  rewriter.setInsertionPoint(recovered.extractValues);

  auto inputLayout = mlir::cast<ttcore::MetalLayoutAttr>(
      mlir::cast<RankedTensorType>(recovered.leaf.getInputs()[0].getType())
          .getEncoding());
  // Non-target slices exchange no data, so a 2D split runs the band-and-merge
  // pipeline once per slice row.
  std::pair<Value, Value> level = emitShardedLeaves(
      rewriter, loc, recovered.logicalInput, k, dim, strategy,
      inputLayout.getMemorySpace(),
      /*gridCols=*/strategy.multiCore ? strategy.numShards : 1,
      /*ntCores=*/strategy.ntShards);

  int64_t bands = strategy.multiCore ? strategy.numShards : 1;
  for (int64_t numGroups : strategy.mergeSchedule) {
    level = emitMergeRound(rewriter, loc, level.first, level.second, k, dim,
                           strategy.outputReductionTiles, bands, numGroups);
    bands = numGroups;
  }

  // A data-parallel split must keep each extract's destination on the same
  // cores as its source; otherwise the conversion's destination still fits.
  auto rebuildExtract = [&](GenericOp oldExtract, Value newInput) -> Value {
    Value output = oldExtract.getOutputs()[0];
    if (strategy.dataParallel) {
      RankedTensorType logicalType = getLogicalType(output);
      llvm::SmallVector<int64_t> tileShape =
          llvm::to_vector(ttcore::TileType::getDefaultShape());
      llvm::SmallVector<int64_t> shardTiles(logicalType.getRank(), 1);
      shardTiles[1 - dim] = strategy.paddedNonTargetTiles;
      auto layout = utils::buildShardedTileLayout(
          rewriter.getContext(), logicalType.getShape(), shardTiles,
          mlir::cast<ttcore::MetalLayoutAttr>(
              mlir::cast<RankedTensorType>(output.getType()).getEncoding())
              .getMemorySpace());
      llvm::SmallVector<int64_t> grid =
          (dim == 1) ? llvm::SmallVector<int64_t>{strategy.ntShards, 1}
                     : llvm::SmallVector<int64_t>{1, strategy.ntShards};
      output =
          rewriter
              .create<EmptyOp>(loc, layout.getDeviceShape(grid, tileShape),
                               ttcore::TileType::get(
                                   logicalType.getElementType(), tileShape),
                               layout)
              .getResult();
    }
    return utils::emitTopKExtract(rewriter, loc, newInput, output,
                                  geometry.extractProjectDim,
                                  strategy.outputReductionTiles, dim)
        ->getResult(0);
  };

  Value newVals = rebuildExtract(recovered.extractValues, level.first);
  Value newIdx = rebuildExtract(recovered.extractIndices, level.second);

  GenericOp lastIdxOp = recovered.extractIndices;
  if (recovered.indexCast) {
    // Elementwise, so it follows its input's layout.
    auto idxType = mlir::cast<RankedTensorType>(newIdx.getType());
    auto castTileType = ttcore::TileType::get(
        getLogicalType(recovered.indexCast->getResult(0)).getElementType(),
        mlir::cast<ttcore::TileType>(idxType.getElementType()).getShape());
    auto castEmpty = rewriter.create<EmptyOp>(
        loc, idxType.getShape(), castTileType, idxType.getEncoding());
    newIdx = utils::emitUnaryGeneric(
        rewriter, loc, newIdx, castEmpty.getResult(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          return b.create<TileTypecastOp>(l, args[1].getType(), args[0])
              .getResult();
        });
    utils::markPinnedGrid(newIdx.getDefiningOp());
    lastIdxOp = recovered.indexCast;
  }

  rewriter.replaceAllUsesWith(recovered.extractValues->getResult(0), newVals);
  rewriter.replaceAllUsesWith(lastIdxOp->getResult(0), newIdx);
  eraseDeadChain(rewriter, {recovered.extractValues.getOperation(),
                            lastIdxOp.getOperation(),
                            recovered.extractIndices.getOperation()});
}

class D2MLowerTopk final : public impl::D2MLowerTopkBase<D2MLowerTopk> {
public:
  using impl::D2MLowerTopkBase<D2MLowerTopk>::D2MLowerTopkBase;

  void runOnOperation() override {
    // Collected up front: the rebuild emits already-sharded leaf topk_blocks of
    // its own. generate_indices distinguishes a leaf from a merge stage.
    llvm::SmallVector<GenericOp> leaves;
    getOperation().walk([&](TopkBlockOp op) {
      GenericOp leaf = op->getParentOfType<GenericOp>();
      if (op.getGenerateIndices() && leaf && leaf.getInputs().size() == 2 &&
          leaf->getNumResults() == 2) {
        leaves.push_back(leaf);
      }
    });
    if (leaves.empty()) {
      return;
    }

    auto workerGrid = llvm::to_vector(
        ttcore::lookupDevice(getOperation()).getWorkerGrid().getShape());
    auto chipDesc =
        ttcore::getCurrentScopeSystemDesc(getOperation()).getChipDesc(0);
    IRRewriter rewriter(&getContext());

    for (GenericOp leaf : leaves) {
      auto topkBlock = *leaf->getRegion(0).getOps<TopkBlockOp>().begin();
      int32_t k = topkBlock.getK();
      int32_t dim = topkBlock.getDim();
      // Taken from the leaf's own operand, since the strategy has to be known
      // before deciding whether the chain around it matters.
      auto inputLayout = mlir::cast<ttcore::MetalLayoutAttr>(
          mlir::cast<RankedTensorType>(leaf.getInputs()[0].getType())
              .getEncoding());

      std::string failureReason;
      auto strategy = selectTopKShardingStrategy(
          k, dim, inputLayout.getLogicalShape(), workerGrid,
          topKL1Budget(leaf, chipDesc,
                       BlockFactorAnalysis::Options{}.numBuffers),
          failureReason);
      if (failed(strategy)) {
        // D2M has no fallback for a topk it cannot place.
        topkBlock->emitOpError(failureReason);
        return signalPassFailure();
      }
      // One core already holds the whole reduction, so ttir-to-d2m's output is
      // final and the chain around it never has to be recovered.
      if (!strategy->multiCore && !strategy->dataParallel) {
        continue;
      }

      auto recovered = recoverSingleCoreTopK(leaf);
      if (failed(recovered)) {
        topkBlock->emitOpError(
            "needs to be split across cores, but the layout and extract chain "
            "around it is not the one ttir-to-d2m emits");
        return signalPassFailure();
      }
      emitMultiCore(rewriter, *recovered, k, dim, *strategy);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
