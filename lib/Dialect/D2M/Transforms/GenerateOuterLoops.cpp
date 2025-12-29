// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERATEOUTERLOOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenerateOuterLoopsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  buildLoopBounds(OpBuilder &builder, Location loc,
                  ArrayRef<int64_t> loopBounds) {
    Value zero = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                   builder.getIndexAttr(0));
    Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                  builder.getIndexAttr(1));
    SmallVector<Value> lbs(loopBounds.size(), zero);
    SmallVector<Value> ubs(llvm::map_range(loopBounds, [&](int64_t dim) {
      return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                               builder.getIndexAttr(dim));
    }));
    SmallVector<Value> step(loopBounds.size(), one);
    return std::make_tuple(lbs, ubs, step);
  }

  static scf::LoopNest buildLoopNest(PatternRewriter &rewriter, Location loc,
                                     ArrayRef<int64_t> loopBounds,
                                     Block *regionBlock, Block *loopedBlock) {
    auto [lbs, ubs, steps] = buildLoopBounds(rewriter, loc, loopBounds);

    return scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange iters) {
          rewriter.setInsertionPointToStart(bodyBuilder.getInsertionBlock());
          Block *innerLoopBlock = bodyBuilder.getInsertionBlock();
          rewriter.mergeBlocks(regionBlock, innerLoopBlock,
                               loopedBlock->getArguments());
        });
  }

  static void replaceIterIndexUses(PatternRewriter &rewriter, Location loc,
                                   scf::LoopNest &loopNest) {
    loopNest.loops.back().walk([&](IterIndexOp index) {
      uint64_t loopDepth = index.getDim();
      assert(loopDepth < loopNest.loops.size());
      scf::ForOp loop = loopNest.loops[loopDepth];
      rewriter.replaceOp(index, loop.getInductionVar());
    });
  }

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Only match GenericOp with a single region (single region/thread form
    // after DMA insertion)
    if (generic.getNumRegions() != 1) {
      return failure();
    }

    SmallVector<int64_t> loopBounds = generic.getLoopBounds();
    if (loopBounds.empty()) {
      // No loops to generate
      return failure();
    }

    // Create a new GenericOp with the same structure
    auto loopedGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(),
        /* block_factors */ rewriter.getArrayAttr({}),
        /* indexing_maps */ rewriter.getArrayAttr({}),
        /* iterator_types */ rewriter.getArrayAttr({}), generic.getThreads(),
        generic.getNumRegions());

    // Process the single region
    Region &region = generic.getRegion(0);
    Block *regionBlock = &region.front();
    Block *loopedBlock = &loopedGeneric.getRegion(0).emplaceBlock();
    loopedBlock->addArguments(
        region.getArgumentTypes(),
        SmallVector<mlir::Location>(region.getArgumentTypes().size(),
                                    generic.getLoc()));
    rewriter.setInsertionPointToStart(loopedBlock);
    scf::LoopNest loopNest = buildLoopNest(
        rewriter, generic.getLoc(), loopBounds, regionBlock, loopedBlock);
    rewriter.modifyOpInPlace(loopedGeneric, [&]() {
      replaceIterIndexUses(rewriter, generic.getLoc(), loopNest);
    });

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
