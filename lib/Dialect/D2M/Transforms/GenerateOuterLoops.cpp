// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

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

    // Create upper bound values using GetBlockFactorOp.
    SmallVector<Value> ubs;
    for (unsigned i = 0; i < numDims; ++i) {
      ubs.push_back(
          rewriter.create<GetBlockFactorOp>(loc, static_cast<int64_t>(i)));
    }

    // Upper bound map: ()[s0] -> (s0).
    AffineMap ubMap = AffineMap::get(0, 1, rewriter.getAffineSymbolExpr(0),
                                     rewriter.getContext());

    // Build nested affine.for loops from outermost to innermost.
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

  static void lowerIndexOps(PatternRewriter &rewriter, Location loc,
                            SmallVector<affine::AffineForOp> &loops,
                            GenericOp generic) {
    SmallVector<Operation *> indexOps;
    generic->walk([&](Operation *op) {
      if (isa<BlockIndexOp, IterIndexOp>(op)) {
        indexOps.push_back(op);
      }
    });

    for (Operation *op : indexOps) {
      int64_t dim = isa<BlockIndexOp>(op) ? cast<BlockIndexOp>(op).getDim()
                                          : cast<IterIndexOp>(op).getDim();
      bool dimInBounds = dim >= 0 && static_cast<size_t>(dim) < loops.size();
      TT_assertv(dimInBounds,
                 "index dim {} out of bounds for loop nest size {}", dim,
                 loops.size());
      Value loopIv = loops[dim].getInductionVar();
      if (auto blockIndex = dyn_cast<BlockIndexOp>(op)) {
        rewriter.setInsertionPoint(blockIndex);
        Value offset = rewriter.create<BlockOffsetOp>(loc, dim);
        Value index = rewriter.create<arith::AddIOp>(loc, loopIv, offset);
        rewriter.replaceOp(blockIndex, index);
        continue;
      }
      rewriter.replaceOp(op, ValueRange(loopIv));
    }
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
        if (auto forOp = mlir::dyn_cast<affine::AffineForOp>(&op)) {
          if (forOp->hasAttr("d2m.blocking_loop")) {
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
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        /* block_factors */ generic.getBlockFactors(),
        /* indexing_maps */ generic.getIndexingMaps(),
        /* iterator_types */ generic.getIteratorTypes(), generic.getThreads(),
        generic.getFabricConnectionConfigAttr(), generic.getNumRegions());

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
    for (auto [i, loop] : llvm::enumerate(loops)) {
      loop->setAttr("d2m.blocking_loop",
                    rewriter.getI64IntegerAttr(static_cast<int64_t>(i)));
      if (generic.getIteratorTypes().size() > i) {
        auto iteratorType =
            mlir::cast<ttcore::IteratorTypeAttr>(generic.getIteratorTypes()[i])
                .getValue();
        if (iteratorType == ttcore::IteratorType::Reduction) {
          loop->setAttr("d2m.reduction_loop", rewriter.getUnitAttr());
        }
      }
    }

    // Lower block_index(dim) -> block_offset(dim) + loop_iv(dim), and
    // iter_index(dim) -> loop_iv(dim).
    rewriter.setInsertionPointToStart(loopedBlock);
    lowerIndexOps(rewriter, generic.getLoc(), loops, loopedGeneric);

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
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
