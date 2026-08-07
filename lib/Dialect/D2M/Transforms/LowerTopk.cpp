// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/TopKShardingStrategy.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/TopKUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERTOPK
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kTileWidth = ttcore::TileType::getDefaultShape()[1];

// Reads the placement plan: every buffer this pass builds comes from here, in
// the order `planTopKBuffers` listed them, and grid selection already chose and
// folded each one.
class PlanReader {
public:
  explicit PlanReader(TopKPlanAttr plan) : plan(plan) {}

  utils::PlacedBuffer next() {
    TT_assertv(cursor < plan.getPlacements().size(),
               "topk emission asked for more buffers than the plan holds; the "
               "plan and the emission have drifted apart");
    TopKPlacementAttr placement = plan.getPlacements()[cursor++];
    return {placement.getType(), placement.getVgmForward(),
            placement.getVgmInverse(), placement.getGrid()};
  }

  /// Every placement must be read by the time emission finishes, or the plan
  /// and the emission have drifted apart.
  void assertFullyConsumed() const {
    TT_assertv(cursor == plan.getPlacements().size(),
               "topk plan holds {} placements but the emission built {}",
               plan.getPlacements().size(), cursor);
  }

private:
  TopKPlanAttr plan;
  std::size_t cursor = 0;
};

/// Narrows every core's shard down to `narrow`'s reduction extent, so the wide
/// partials do not stay live on every core and overflow L1.
Value compactReductionDim(RewriterBase &rewriter, Location loc, Value partial,
                          const utils::PlacedBuffer &narrow) {
  auto wideType = mlir::cast<RankedTensorType>(partial.getType());
  auto tileType = mlir::cast<ttcore::TileType>(wideType.getElementType());
  const std::size_t rank = wideType.getShape().size() / 2;

  Value narrowEmpty = rewriter
                          .create<EmptyOp>(loc, narrow.type, narrow.vgmInverse,
                                           narrow.vgmForward)
                          .getResult();
  auto generic = rewriter.create<GenericOp>(
      loc, TypeRange{narrow.type}, ValueRange{partial}, ValueRange{narrowEmpty},
      /*additionalArgs=*/ValueRange(), narrow.grid,
      /*block_factors=*/rewriter.getI64ArrayAttr({}),
      /*indexing_maps=*/rewriter.getArrayAttr({}),
      /*iterator_types=*/rewriter.getArrayAttr({}),
      rewriter.getArrayAttr(rewriter.getAttr<ThreadAttr>(ThreadType::Unified)),
      /*fabricConnectionConfig=*/nullptr, /*numRegions=*/1);

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.createBlock(&generic->getRegion(0));
    auto shardShape = [&](RankedTensorType deviceType) {
      return deviceType.getShape().take_back(rank);
    };
    llvm::SmallVector<Value> coreIndices;
    coreIndices.reserve(rank);
    for (std::size_t i = 0; i < rank; ++i) {
      coreIndices.push_back(
          rewriter.create<CoreIndexOp>(loc, static_cast<int64_t>(i)));
    }
    auto loadBuf =
        rewriter.create<tensor::EmptyOp>(loc, shardShape(wideType), tileType);
    Value loaded =
        rewriter
            .create<RemoteLoadOp>(
                loc, RankedTensorType::get(shardShape(wideType), tileType),
                loadBuf, partial, coreIndices)
            .getResult();
    auto outBuf = rewriter.create<tensor::EmptyOp>(loc, shardShape(narrow.type),
                                                   tileType);
    AffineMap shardIdentity = rewriter.getMultiDimIdentityMap(rank);
    Value narrowed =
        rewriter
            .create<LocalCopyOp>(
                loc, loaded, outBuf.getResult(),
                rewriter.getAffineMapArrayAttr({shardIdentity, shardIdentity}))
            .getResult();
    Value stored = rewriter
                       .create<RemoteStoreOp>(loc, narrow.type, narrowEmpty,
                                              coreIndices, narrowed)
                       .getResult();
    rewriter.create<YieldOp>(loc, ValueRange{stored});
  }

  return utils::materializeToLayout(rewriter, loc, generic->getResult(0),
                                    narrow);
}

/// Collapses one merge round: each surviving core merges the bands gathered
/// through a composite view (resolved into DMA reads by
/// D2MExpandDMAReadCompositeView).
std::pair<Value, Value> emitMergeRound(RewriterBase &rewriter, Location loc,
                                       Value valsIn, Value idxIn, int32_t k,
                                       int32_t dim, PlanReader &plan) {
  MLIRContext *ctx = rewriter.getContext();
  auto valsInType = mlir::cast<RankedTensorType>(valsIn.getType());
  const std::size_t rank = valsInType.getShape().size() / 2;
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
  llvm::SmallVector<Attribute> iterators(
      rank, ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

  // Separate generics: the DMA-expansion pass supports only one composite view
  // per GenericOp (#7600).
  auto gather = [&](Value in) -> Value {
    utils::PlacedBuffer wide = plan.next();
    auto composite = rewriter.create<CompositeViewOp>(
        loc, wide.type, ValueRange{in}, dim, /*logicalSizes=*/nullptr);
    llvm::SmallVector<Value> ins = {composite.getResult()};
    llvm::SmallVector<Value> outs = {
        rewriter
            .create<EmptyOp>(loc, wide.type, wide.vgmInverse, wide.vgmForward)
            .getResult()};
    auto generic = rewriter.create<GenericOp>(
        loc, ins, outs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(
            llvm::SmallVector<AffineMap>{identityMap, identityMap}),
        rewriter.getArrayAttr(iterators), ThreadType::Unified, wide.grid);
    utils::buildParallelGenericRegion(
        rewriter, loc, generic, ins, outs,
        [](ArrayRef<Value> args) -> llvm::SmallVector<Value> {
          return {args[0]};
        });
    return utils::materializeToLayout(rewriter, loc, generic->getResult(0),
                                      wide);
  };

  // Braced init so the two gathers draw from the plan in order.
  llvm::SmallVector<Value> ins = {gather(valsIn), gather(idxIn)};

  utils::PlacedBuffer mergeVals = plan.next();
  utils::PlacedBuffer mergeIdx = plan.next();
  llvm::SmallVector<Value> outs = {
      rewriter
          .create<EmptyOp>(loc, mergeVals.type, mergeVals.vgmInverse,
                           mergeVals.vgmForward)
          .getResult(),
      rewriter
          .create<EmptyOp>(loc, mergeIdx.type, mergeIdx.vgmInverse,
                           mergeIdx.vgmForward)
          .getResult()};

  const int64_t numElements =
      mergeVals.type.getShape()[rank + dim] * kTileWidth;

  auto merge = rewriter.create<GenericOp>(
      loc, ins, outs, /*additionalArgs=*/ValueRange(),
      rewriter.getAffineMapArrayAttr(llvm::SmallVector<AffineMap>{
          identityMap, identityMap, identityMap, identityMap}),
      rewriter.getArrayAttr(iterators), ThreadType::Unified, mergeVals.grid);
  utils::buildParallelGenericRegion(
      rewriter, loc, merge, ins, outs,
      [&](ArrayRef<Value> args) -> llvm::SmallVector<Value> {
        // No generate_indices: these indices are an earlier stage's results and
        // cannot be recomputed from a coordinate.
        auto block = rewriter.create<TopkBlockOp>(
            loc, args[0], args[1], args[2], args[3], k, numElements,
            /*stableSort=*/false, dim);
        return {block.getResultValues(), block.getResultIndices()};
      });

  utils::PlacedBuffer compactVals = plan.next();
  utils::PlacedBuffer compactIdx = plan.next();
  return {compactReductionDim(rewriter, loc,
                              utils::materializeToLayout(rewriter, loc,
                                                         merge->getResult(0),
                                                         mergeVals),
                              compactVals),
          compactReductionDim(rewriter, loc,
                              utils::materializeToLayout(
                                  rewriter, loc, merge->getResult(1), mergeIdx),
                              compactIdx)};
}

/// Collapses the reduction dim to tile 0, undoing what `emitLeafTopk` did.
/// `outputReductionTiles` is ceil(k / 32).
GenericOp emitTopKExtract(RewriterBase &rewriter, Location loc,
                          Value topkResult, Value extractOutput, int32_t dim,
                          int64_t outputReductionTiles, ttcore::GridAttr grid) {
  const auto extractProjectDim = static_cast<std::size_t>(dim);
  MLIRContext *ctx = rewriter.getContext();
  std::size_t extractRank =
      ttcore::getDeviceLayout(extractOutput).getRank() / 2;
  AffineMap extractIdentity = rewriter.getMultiDimIdentityMap(extractRank);
  llvm::SmallVector<Attribute> extractIters(
      extractRank,
      ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

  llvm::SmallVector<AffineExpr> inputMapExprs;
  for (std::size_t i = 0; i < extractRank; ++i) {
    inputMapExprs.push_back(i == extractProjectDim
                                ? rewriter.getAffineConstantExpr(0)
                                : rewriter.getAffineDimExpr(i));
  }
  AffineMap inputProjectedMap =
      AffineMap::get(extractRank, 0, inputMapExprs, ctx);

  llvm::SmallVector<Value> extractInputs = {topkResult};
  llvm::SmallVector<Value> extractOutputs = {extractOutput};
  auto generic = rewriter.create<GenericOp>(
      loc, extractInputs, extractOutputs, /*additionalArgs=*/ValueRange(),
      rewriter.getAffineMapArrayAttr(
          llvm::SmallVector<AffineMap>{inputProjectedMap, extractIdentity}),
      rewriter.getArrayAttr(extractIters), ThreadType::Unified, grid);

  utils::buildParallelGenericRegion(
      rewriter, loc, generic, extractInputs, extractOutputs,
      [&](ArrayRef<Value> blockArgs) -> llvm::SmallVector<Value> {
        Value input = blockArgs[0];
        Value output = blockArgs[1];
        std::size_t outShardRank =
            cast<RankedTensorType>(output.getType()).getRank();
        int64_t inReductionExtent = cast<RankedTensorType>(input.getType())
                                        .getShape()[extractProjectDim];
        AffineExpr projExpr =
            outputReductionTiles == 1
                ? rewriter.getAffineConstantExpr(0)
                : rewriter.getAffineDimExpr(extractProjectDim) %
                      inReductionExtent;
        llvm::SmallVector<AffineExpr> mapFirstExprs;
        for (std::size_t i = 0; i < outShardRank; ++i) {
          mapFirstExprs.push_back(
              i == extractProjectDim ? projExpr : rewriter.getAffineDimExpr(i));
        }
        AffineMap mapFirst =
            AffineMap::get(outShardRank, 0, mapFirstExprs, ctx);
        AffineMap outIdentity = rewriter.getMultiDimIdentityMap(outShardRank);
        llvm::SmallVector<mlir::utils::IteratorType> linalgIters(
            outShardRank, mlir::utils::IteratorType::parallel);

        auto linalgOp = rewriter.create<linalg::GenericOp>(
            loc, output.getType(), input, output,
            llvm::SmallVector<AffineMap>{mapFirst, outIdentity}, linalgIters,
            [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
              Value result;
              if (dim == 1) {
                result = b.create<TileTransposeOp>(bodyLoc, args[1].getType(),
                                                   args[0]);
              } else {
                result = b.create<TileTypecastOp>(bodyLoc, args[1].getType(),
                                                  args[0]);
              }
              b.create<linalg::YieldOp>(bodyLoc, result);
            });
        return {linalgOp->getResult(0)};
      });

  return generic;
}

/// The laid-out and masked input d2m-grid-selection emitted off `logicalInput`,
/// marked because the placeholder leaf cannot hold it as an operand.
Value findPlacedInput(Value logicalInput) {
  // The mark sits on the to_layout, or on the mask when one was planned, and
  // is dropped on the way out so no marker outlives the pass.
  auto take = [](Operation *op) -> Value {
    if (!op->hasAttr(utils::kTopKInputAttr)) {
      return {};
    }
    op->removeAttr(utils::kTopKInputAttr);
    return op->getResult(0);
  };
  for (Operation *toLayout : logicalInput.getUsers()) {
    if (Value placed = take(toLayout)) {
      return placed;
    }
    for (Operation *mask : toLayout->getUsers()) {
      if (Value placed = take(mask)) {
        return placed;
      }
    }
  }
  TT_assertv(false, "topk leaf has no input placed by d2m-grid-selection");
  return {};
}

/// Builds the topk the plan describes, replacing the leaf d2m-grid-selection
/// placed. `chain` comes by value because MLIR op accessors are non-const.
void emitPlannedTopK(RewriterBase &rewriter, SingleCoreTopK chain, int32_t k,
                     int32_t dim, TopKPlanAttr planAttr) {
  Location loc = chain.leaf->getLoc();
  PlanReader plan(planAttr);
  auto logicalInputType =
      mlir::cast<RankedTensorType>(chain.logicalInput.getType());

  // The only thing the build reads is the input chain grid selection laid out
  // and masked onto the split; everything else it re-emits from the plan.
  rewriter.setInsertionPoint(chain.leaf);
  Value input = findPlacedInput(chain.logicalInput);

  utils::LeafTopKBuffers leafBuffers;
  if (dim == 1) {
    leafBuffers.transpose = plan.next();
  }
  leafBuffers.scratch = plan.next();
  leafBuffers.values = plan.next();
  leafBuffers.indices = plan.next();
  auto [leafVals, leafIdx] =
      utils::emitLeafTopk(rewriter, loc, input, k, dim,
                          logicalInputType.getShape()[dim], leafBuffers);

  // Banding is what produces wide partials worth narrowing; a purely
  // data-parallel split leaves each core's result already final.
  std::pair<Value, Value> level;
  if (planAttr.getNumShards() > 1) {
    utils::PlacedBuffer compactVals = plan.next();
    utils::PlacedBuffer compactIdx = plan.next();
    level = {compactReductionDim(rewriter, loc, leafVals, compactVals),
             compactReductionDim(rewriter, loc, leafIdx, compactIdx)};
  } else {
    level = {
        utils::materializeToLayout(rewriter, loc, leafVals, leafBuffers.values),
        utils::materializeToLayout(rewriter, loc, leafIdx,
                                   leafBuffers.indices)};
  }

  for (std::size_t round = 0; round < planAttr.getMergeSchedule().size();
       ++round) {
    level =
        emitMergeRound(rewriter, loc, level.first, level.second, k, dim, plan);
  }

  auto emitExtract = [&](Value newInput) -> Value {
    utils::PlacedBuffer placed = plan.next();
    // Width comes from `k`, not the input's reduction extent: a data-parallel
    // split has no merge tree to narrow it, so the partial is still full width.
    return emitTopKExtract(rewriter, loc, newInput,
                           rewriter
                               .create<EmptyOp>(loc, placed.type,
                                                placed.vgmInverse,
                                                placed.vgmForward)
                               .getResult(),
                           dim, llvm::divideCeil(k, kTileWidth), placed.grid)
        ->getResult(0);
  };

  Value newVals = emitExtract(level.first);
  Value newIdx = emitExtract(level.second);

  // dim == 0's extract casts to the user's index type on the way out; dim == 1
  // transposes instead, so its cast needs a region of its own.
  if (dim != 0) {
    utils::PlacedBuffer castBuffer = plan.next();
    newIdx = utils::emitUnaryGeneric(
        rewriter, loc, newIdx,
        rewriter
            .create<EmptyOp>(loc, castBuffer.type, castBuffer.vgmInverse,
                             castBuffer.vgmForward)
            .getResult(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          return b.create<TileTypecastOp>(l, args[1].getType(), args[0])
              .getResult();
        },
        castBuffer.grid);
  }

  plan.assertFullyConsumed();

  // The placed leaf's results feed nothing but the `to_layout`s that hand the
  // topk back to its user, and those check neither shape nor element type.
  rewriter.replaceAllUsesWith(chain.leaf->getResult(0), newVals);
  rewriter.replaceAllUsesWith(chain.leaf->getResult(1), newIdx);
  rewriter.eraseOp(chain.leaf);
}

class D2MLowerTopk final : public impl::D2MLowerTopkBase<D2MLowerTopk> {
public:
  using impl::D2MLowerTopkBase<D2MLowerTopk>::D2MLowerTopkBase;

  void runOnOperation() override {
    // Collected up front because the build emits generics of its own; only the
    // leaf d2m-grid-selection placed carries a plan.
    llvm::SmallVector<GenericOp> leaves;
    getOperation().walk([&](GenericOp generic) {
      if (generic->hasAttr(utils::kTopKPlanAttr)) {
        leaves.push_back(generic);
      }
    });

    IRRewriter rewriter(&getContext());
    for (GenericOp leaf : leaves) {
      auto topkBlock = *leaf->getRegion(0).getOps<TopkBlockOp>().begin();
      auto plan = leaf->getAttrOfType<TopKPlanAttr>(utils::kTopKPlanAttr);

      emitPlannedTopK(rewriter, readSingleCoreTopK(leaf), topkBlock.getK(),
                      topkBlock.getDim(), plan);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
