// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_WRAPSINGLEAFFINELOOPS
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

/// Pattern to wrap single (top-level) AffineFor operations with an outer
/// dummy AffineFor loop that goes from 0 to 1.
class WrapSingleAffineLoopPattern
    : public OpRewritePattern<affine::AffineForOp> {
public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Skip if the loop is already nested inside another AffineFor.
    if (forOp->getParentOfType<affine::AffineForOp>()) {
      return failure();
    }

    // Skip if the loop is not inside a function.
    if (!forOp->getParentOfType<func::FuncOp>()) {
      return failure();
    }

    // Skip if the loop has any nested loops.
    if (!forOp.getBody()->getOps<affine::AffineForOp>().empty()) {
      return failure();
    }

    Location loc = forOp.getLoc();

    // Create the wrapper affine for loop.
    auto wrapperLoop = rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/0, /*upperBound=*/1, /*step=*/1);

    Block *wrapperBlock = wrapperLoop.getBody();
    rewriter.setInsertionPointToStart(wrapperBlock);

    rewriter.clone(*forOp);

    rewriter.replaceOp(forOp, wrapperLoop);

    return success();
  }
};

class WrapSingleAffineLoops
    : public impl::WrapSingleAffineLoopsBase<WrapSingleAffineLoops> {
public:
  using impl::WrapSingleAffineLoopsBase<
      WrapSingleAffineLoops>::WrapSingleAffineLoopsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<WrapSingleAffineLoopPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::transforms
