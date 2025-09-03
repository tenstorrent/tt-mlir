// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/PassManager.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_WRAPSINGLEAFFINELOOPS
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

/// Pattern to wrap single (top-level) AffineFor operations with an
/// outerAffineFor loops that are specialized for gpu use.
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

    int64_t upperBoundValueInt = 1;
    int64_t originalUpperBound = 1;
    int64_t blockSize = 1;
    int64_t threadSize = 1;
    Location loc = forOp.getLoc();
    auto upperBound = forOp.getUpperBound().getMap().getResult(0);
    auto upperBoundValue = dyn_cast<AffineConstantExpr>(upperBound);
    if (upperBoundValue) {
      originalUpperBound = upperBoundValue.getValue();
      upperBoundValueInt = upperBoundValue.getValue();
      while (blockSize * blockSize < upperBoundValueInt && blockSize < 1024) {
        blockSize *= 2;
      }

      upperBoundValueInt = (upperBoundValueInt + blockSize - 1) / blockSize;
      if (upperBoundValueInt > 1024) {
        while (threadSize < upperBoundValueInt && threadSize < 1024) {
          threadSize *= 2;
        }
        upperBoundValueInt = (upperBoundValueInt + threadSize - 1) / threadSize;
      }
      AffineMap newUpperBound =
          AffineMap::getConstantMap(upperBoundValueInt, rewriter.getContext());
      forOp.setUpperBound(ValueRange{}, newUpperBound);
    }

    auto blockLoop = rewriter.create<affine::AffineForOp>(
        loc, /*lowerBound=*/0, /*upperBound=*/blockSize, /*step=*/1);

    Block *blockBlock = blockLoop.getBody();
    rewriter.setInsertionPointToStart(blockBlock);

    affine::AffineForOp threadLoop;
    if (threadSize > 1) {
      threadLoop = rewriter.create<affine::AffineForOp>(
          loc, /*lowerBound=*/0, /*upperBound=*/threadSize, /*step=*/1);
      Block *threadBlock = threadLoop.getBody();
      rewriter.setInsertionPointToStart(threadBlock);
    }

    auto clonedForOp = rewriter.clone(*forOp);
    auto clonedAffineFor = cast<affine::AffineForOp>(clonedForOp);

    auto blockIv = blockLoop.getInductionVar();
    auto originalIv = clonedAffineFor.getInductionVar();
    rewriter.setInsertionPointToStart(clonedAffineFor.getBody());

    affine::AffineApplyOp newIv;
    if (threadSize > 1) {
      auto threadIv = threadLoop.getInductionVar();
      // New index: block_iterator * threadSize * remainder + thread_iterator *
      // remainder + remainder_iterator.
      auto blockThreadRemainder = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(
              1, 0,
              {rewriter.getAffineDimExpr(0) * threadSize * upperBoundValueInt},
              rewriter.getContext()),
          blockIv);
      auto threadRemainder = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(1, 0,
                         {rewriter.getAffineDimExpr(0) * upperBoundValueInt},
                         rewriter.getContext()),
          threadIv);
      auto intermediateIv = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(
              2, 0,
              {rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1)},
              rewriter.getContext()),
          ValueRange{blockThreadRemainder, threadRemainder});
      newIv = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(
              2, 0,
              {rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1)},
              rewriter.getContext()),
          ValueRange{intermediateIv, originalIv});
    } else {
      // New index: block_iterator * remainder + remainder_iterator.
      auto blockRemainder = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(1, 0,
                         {rewriter.getAffineDimExpr(0) * upperBoundValueInt},
                         rewriter.getContext()),
          blockIv);
      newIv = rewriter.create<affine::AffineApplyOp>(
          loc,
          AffineMap::get(
              2, 0,
              {rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1)},
              rewriter.getContext()),
          ValueRange{blockRemainder, originalIv});
    }
    // Create a guard condition to skip iterations when the computed index
    // exceeds the original bounds.
    auto originalBoundConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIndexAttr(originalUpperBound));

    auto condition = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, newIv, originalBoundConst);

    auto ifOp = rewriter.create<scf::IfOp>(loc, condition, /*hasElse=*/false);

    Block *thenBlock = &ifOp.getThenRegion().front();
    rewriter.setInsertionPointToStart(thenBlock);
    // Dummy operation to know where to move the body of the innermost loop.
    Operation *lastOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIndexAttr(originalUpperBound));
    bool passedIf = false;
    SmallVector<Operation *> operationsToMove;
    for (auto &op : clonedAffineFor.getBody()->getOperations()) {
      if (isa<scf::IfOp>(op)) {
        passedIf = true;
        continue;
      }
      if (!passedIf) {
        continue;
      }
      if (!isa<affine::AffineYieldOp>(op)) {
        operationsToMove.push_back(&op);
      }
    }
    for (auto &op : operationsToMove) {
      rewriter.moveOpAfter(op, lastOp);
      lastOp = op;
    }

    originalIv.replaceUsesWithIf(newIv, [&](OpOperand &operand) {
      // Skip if this operand belongs to the defining operation of originalIv.
      if (operand.getOwner() == originalIv.getDefiningOp()) {
        return false;
      }

      // Skip if this operand belongs to newIv (used to compute newIv).
      if (operand.getOwner() == newIv) {
        return false;
      }
      return true;
    });

    rewriter.replaceOp(forOp, blockLoop);
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
