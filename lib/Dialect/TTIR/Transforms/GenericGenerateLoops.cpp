// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GenericInterchangeAnalysis.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICGENERATELOOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericGenerateLoopsRewriter : public OpRewritePattern<GenericOp> {
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

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getIndexingMaps().empty()) {
      return failure();
    }

    auto loopedGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(), rewriter.getArrayAttr({}),
        rewriter.getArrayAttr({}), generic.getNumRegions());

    SmallVector<int64_t> interchange = calculateOptimalInterchange(generic);

    SmallVector<int64_t> ones(interchange.size(), 1);
    for (Region& region : generic.getRegions()) {
      Block *regionBlock = &region.front();
      Block *loopedBlock =
          &loopedGeneric.getRegion(region.getRegionNumber()).emplaceBlock();
      loopedBlock->addArguments(
          region.getArgumentTypes(),
          SmallVector<mlir::Location>(region.getArgumentTypes().size(),
                                      generic.getLoc()));
      rewriter.setInsertionPointToStart(loopedBlock);
      buildLoopNest(rewriter, generic.getLoc(), ones,
                                             regionBlock, loopedBlock);
    }

    rewriter.replaceOp(generic, loopedGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class TTIRGenericGenerateLoops
    : public impl::TTIRGenericGenerateLoopsBase<TTIRGenericGenerateLoops> {
public:
  using impl::TTIRGenericGenerateLoopsBase<
      TTIRGenericGenerateLoops>::TTIRGenericGenerateLoopsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericGenerateLoopsRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::ttir
