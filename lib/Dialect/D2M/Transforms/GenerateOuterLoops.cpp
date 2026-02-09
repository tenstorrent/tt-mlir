// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERATEOUTERLOOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenerateOuterLoopsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static SmallVector<affine::AffineForOp>
  buildLoopNest(PatternRewriter &rewriter, Location loc, unsigned numDims,
                Block *regionBlock, Block *loopedBlock) {
    SmallVector<affine::AffineForOp> loops;

    // Create upper bound values using GetBlockFactorOp
    SmallVector<Value> ubs;
    for (unsigned i = 0; i < numDims; ++i) {
      ubs.push_back(
          rewriter.create<GetBlockFactorOp>(loc, static_cast<int64_t>(i)));
    }

    // Upper bound map: ()[s0] -> (s0)
    AffineMap ubMap = AffineMap::get(0, 1, rewriter.getAffineSymbolExpr(0),
                                     rewriter.getContext());

    // Build nested affine.for loops from outermost to innermost
    for (unsigned i = 0; i < numDims; ++i) {
      auto forOp = affine::AffineForOp::create(
          rewriter, loc,
          /*lbOperands=*/{}, /*lbMap=*/rewriter.getConstantAffineMap(0),
          /*ubOperands=*/ValueRange{ubs[i]}, /*ubMap=*/ubMap,
          /*step=*/1);
      loops.push_back(forOp);
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    // Merge the original region block into the innermost loop body.
    // Remove the auto-generated affine.yield terminator first, then merge
    // (which splices to end of block), then re-add the yield.
    Block *innerBody = loops.back().getBody();
    rewriter.eraseOp(innerBody->getTerminator());
    rewriter.mergeBlocks(regionBlock, innerBody, loopedBlock->getArguments());
    rewriter.setInsertionPointToEnd(innerBody);
    rewriter.create<affine::AffineYieldOp>(loc);

    return loops;
  }

  static void replaceIndexOpUses(PatternRewriter &rewriter, Location loc,
                                 SmallVector<affine::AffineForOp> &loops,
                                 GenericOp generic) {
    // Get the output operand indexing map to determine which dimensions
    // participate in the grid
    unsigned outputOperandsIndex = generic.getOutputs().getBeginOperandIndex();
    AffineMap outputOperandIndexingMap =
        mlir::cast<AffineMapAttr>(
            generic.getIndexingMaps()[outputOperandsIndex])
            .getValue();

    // Get the grid mapping for use with CoreIndexOp. The mapping includes
    // a leading device index result, so we use the full mapping and let
    // CoreIndexOp handle dimension selection via (dim + 1).
    AffineMap gridMapping = generic.getGrid().getMapping();

    // The number of grid dimensions (typically 2 for a 2D grid, but could be
    // more with virtualization)
    constexpr unsigned numPhysicalGridDims = 2;
    unsigned numGridDims = gridMapping.isEmpty()
                               ? numPhysicalGridDims
                               : gridMapping.getNumResults() - 1;

    // Create CoreIndexOp operations lazily - create them the first time we need
    // them, at the start of the outermost loop body, then reuse them.
    SmallVector<Value> virtualGridIndices;
    bool virtualGridIndicesCreated = false;

    // Handle BlockIndexOp: apply full grid/block calculation
    loops.back().walk([&](BlockIndexOp index) {
      uint64_t dim = index.getDim();
      assert(dim < loops.size());
      affine::AffineForOp loop = loops[dim];
      Value iterIndex = loop.getInductionVar();

      // Set insertion point to before the BlockIndexOp so we can create
      // operations that will be used to replace it
      rewriter.setInsertionPoint(index);

      // Create CoreIndexOp operations lazily at the start of the outermost loop
      // body if we haven't created them yet. Use CoreIndexOp with the grid
      // mapping directly - the lowering will handle applying the affine map.
      if (!virtualGridIndicesCreated && !loops.empty()) {
        // Set insertion point to the start of the outermost loop body
        rewriter.setInsertionPointToStart(loops.front().getBody());
        virtualGridIndices.resize(numGridDims);
        for (unsigned gridDim = 0; gridDim < numGridDims; gridDim++) {
          virtualGridIndices[gridDim] = rewriter.create<CoreIndexOp>(
              loc, static_cast<int64_t>(gridDim), gridMapping);
        }
        virtualGridIndicesCreated = true;
        // Reset insertion point back to before the BlockIndexOp
        rewriter.setInsertionPoint(index);
      }

      // Check if this iteration dimension maps to a grid dimension in the
      // output operand indexing map. The output operand indexing map maps
      // iteration dimensions to output dimensions. We need to check if the
      // expression corresponding to iteration dimension `dim` appears as one of
      // the first numGridDims results in the output map.
      //
      // Create the expression for iteration dimension `dim` (e.g., d0, d1, d2)
      AffineExpr dimExpr =
          getAffineDimExpr(dim, outputOperandIndexingMap.getContext());

      // Check if this expression appears in the output operand indexing map
      // results
      std::optional<unsigned> gridResult =
          outputOperandIndexingMap.getResultPosition(dimExpr);

      // If the result position exists and is less than numGridDims, then this
      // dimension participates in the grid
      if (gridResult.has_value() && gridResult.value() < numGridDims) {
        // This dimension participates in the grid. Compute:
        // gridIndex * blockFactor + iterIndex
        const unsigned gridDim = gridResult.value();
        assert(gridDim < virtualGridIndices.size() &&
               "Grid dimension index out of bounds");

        Value gridIndex = virtualGridIndices[gridDim];
        Value blockFactor =
            rewriter.create<GetBlockFactorOp>(loc, static_cast<int64_t>(dim));

        Value gridScaled = rewriter.create<arith::MulIOp>(
            loc, rewriter.getIndexType(), gridIndex, blockFactor);
        Value combinedIndex = rewriter.create<arith::AddIOp>(
            loc, rewriter.getIndexType(), gridScaled, iterIndex);

        rewriter.replaceOp(index, combinedIndex);
      } else {
        // This dimension does not participate in the grid, just use the loop
        // induction variable
        rewriter.replaceOp(index, iterIndex);
      }
    });

    // Handle IterIndexOp: simple replacement with loop induction variable
    loops.back().walk([&](IterIndexOp index) {
      uint64_t dim = index.getDim();
      assert(dim < loops.size());
      affine::AffineForOp loop = loops[dim];
      Value iterIndex = loop.getInductionVar();

      rewriter.setInsertionPoint(index);
      rewriter.replaceOp(index, iterIndex);
    });
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Skip explicit datamovement form - users manage loops manually
    if (generic.isExplicitDatamovementForm()) {
      return failure();
    }

    // Only match GenericOp with a single region (single region/thread form
    // after DMA insertion)
    if (generic.getNumRegions() != 1) {
      return failure();
    }

    unsigned numDims =
        static_cast<unsigned>(generic.getBlockFactorsValue().size());
    if (numDims == 0) {
      // No loops to generate
      return failure();
    }

    // Check if loops are already generated (avoid infinite loop)
    // Look for the marker attribute on the outermost loop
    Region &checkRegion = generic.getRegion(0);
    if (!checkRegion.empty()) {
      Block &checkBlock = checkRegion.front();
      for (Operation &op : checkBlock.getOperations()) {
        if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
          if (forOp->hasAttr("d2m.outer_loop")) {
            return failure();
          }
        }
      }
    }

    // Create a new GenericOp with the same structure
    // After generating loops, preserve all attributes including block_factors
    // (needed by LowerLoadStoreOpsToDMA for stream index computation).
    auto loopedGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(),
        /* block_factors */ generic.getBlockFactors(),
        /* indexing_maps */ generic.getIndexingMaps(),
        /* iterator_types */ generic.getIteratorTypes(), generic.getThreads(),
        generic.getScratchInputsAttr(), generic.getNumRegions());

    // Process the single region
    Region &region = generic.getRegion(0);
    Block *regionBlock = &region.front();
    Block *loopedBlock = &loopedGeneric.getRegion(0).emplaceBlock();
    loopedBlock->addArguments(
        region.getArgumentTypes(),
        SmallVector<mlir::Location>(region.getArgumentTypes().size(),
                                    generic.getLoc()));
    rewriter.setInsertionPointToStart(loopedBlock);
    SmallVector<affine::AffineForOp> loops = buildLoopNest(
        rewriter, generic.getLoc(), numDims, regionBlock, loopedBlock);

    // Mark all loops in the nest with an attribute to prevent re-processing.
    // These are called "outer loops" because they wrap the generic operation,
    // iterating over its block factors. The generic operation's regions contain
    // the "inner" computation that executes within each loop iteration.
    for (affine::AffineForOp loop : loops) {
      loop->setAttr("d2m.outer_loop", rewriter.getUnitAttr());
    }

    // Replace IterIndexOp uses. We need to do this after the loops are created
    // but before we replace the generic op, so the operations are created in
    // the right place. Set insertion point back to the start of loopedBlock
    // so CoreIndexOp operations are created in the right place.
    rewriter.setInsertionPointToStart(loopedBlock);
    replaceIndexOpUses(rewriter, generic.getLoc(), loops, loopedGeneric);

    rewriter.replaceOp(generic, loopedGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MGenerateOuterLoops
    : public impl::D2MGenerateOuterLoopsBase<D2MGenerateOuterLoops> {
public:
  using impl::D2MGenerateOuterLoopsBase<
      D2MGenerateOuterLoops>::D2MGenerateOuterLoopsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenerateOuterLoopsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
