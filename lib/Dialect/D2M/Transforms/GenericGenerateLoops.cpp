// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICGENERATELOOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenericGenerateLoopsRewriter : public OpRewritePattern<GenericOp> {
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
    if (generic.getIndexingMaps().empty()) {
      return failure();
    }

    auto loopedGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(), generic.getBlockFactors(),
        /* indexing_maps */ rewriter.getArrayAttr({}),
        /* iterator_types */ rewriter.getArrayAttr({}), generic.getThreads(),
        generic.getNumRegions());

    SmallVector<int64_t> loopBounds = generic.getLoopBounds();
    for (Region &region : generic.getRegions()) {
      Block *regionBlock = &region.front();
      Block *loopedBlock =
          &loopedGeneric.getRegion(region.getRegionNumber()).emplaceBlock();
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
    }

    rewriter.replaceOp(generic, loopedGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MGenericGenerateLoops
    : public impl::D2MGenericGenerateLoopsBase<D2MGenericGenerateLoops> {
public:
  using impl::D2MGenericGenerateLoopsBase<
      D2MGenericGenerateLoops>::D2MGenericGenerateLoopsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericGenerateLoopsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
