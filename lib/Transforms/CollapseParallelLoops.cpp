// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_COLLAPSEPARALLELLOOPS
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

/// Pattern to collapse the last two dimensions in 4D SCF parallel loops.
class CollapseParallelLoopPattern : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    auto lowerBounds = parallelOp.getLowerBound();
    auto upperBounds = parallelOp.getUpperBound();
    auto steps = parallelOp.getStep();

    if (lowerBounds.size() != 4) {
      return failure();
    }

    if (!isCollapsible(lowerBounds[2], upperBounds[2], steps[2]) ||
        !isCollapsible(lowerBounds[3], upperBounds[3], steps[3])) {
      return failure();
    }

    unsigned startIdx = 2;
    unsigned endIdx = 3;
    Location loc = parallelOp.getLoc();
    auto initVals = parallelOp.getInitVals();
    auto inductionVars = parallelOp.getInductionVars();

    SmallVector<int64_t> dimSizes;
    int64_t collapsedSize = 1;

    for (unsigned i = startIdx; i <= endIdx; ++i) {
      auto upperConstant = getConstantIntValue(upperBounds[i]);
      auto lowerConstant = getConstantIntValue(lowerBounds[i]);
      if (!upperConstant || !lowerConstant) {
        return failure();
      }
      int64_t dimSize = *upperConstant - *lowerConstant;
      dimSizes.push_back(dimSize);
      collapsedSize *= dimSize;
    }

    SmallVector<Value> newLowerBounds, newUpperBounds, newSteps;

    newLowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    newUpperBounds.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, collapsedSize));
    newSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));

    newLowerBounds.push_back(lowerBounds[1]);
    newUpperBounds.push_back(upperBounds[1]);
    newSteps.push_back(steps[1]);

    newLowerBounds.push_back(lowerBounds[0]);
    newUpperBounds.push_back(upperBounds[0]);
    newSteps.push_back(steps[0]);

    auto newParallelOp = rewriter.create<scf::ParallelOp>(
        loc, newLowerBounds, newUpperBounds, newSteps, initVals);

    Block *newBody = newParallelOp.getBody();
    Block *oldBody = parallelOp.getBody();

    newBody->clear();

    rewriter.setInsertionPointToStart(newBody);

    SmallVector<Value> newInductionVars = newParallelOp.getInductionVars();
    IRMapping mapping;

    mapping.map(inductionVars[0], newInductionVars[2]);

    mapping.map(inductionVars[1], newInductionVars[1]);

    Value collapsedVar = newInductionVars[0];
    SmallVector<Value> decomposedVars;

    Value remaining = collapsedVar;
    for (unsigned i = startIdx; i <= endIdx; ++i) {
      if (i == endIdx) {

        decomposedVars.push_back(remaining);
      } else {

        int64_t productOfRemainingDims = 1;
        for (unsigned j = i + 1; j <= endIdx; ++j) {
          productOfRemainingDims *= dimSizes[j - startIdx];
        }

        Value divisor = rewriter.create<arith::ConstantIndexOp>(
            loc, productOfRemainingDims);
        Value quotient =
            rewriter.create<arith::DivUIOp>(loc, remaining, divisor);
        decomposedVars.push_back(quotient);

        Value product = rewriter.create<arith::MulIOp>(loc, quotient, divisor);
        remaining = rewriter.create<arith::SubIOp>(loc, remaining, product);
      }
    }
    mapping.map(inductionVars[2], decomposedVars[0]);
    mapping.map(inductionVars[3], decomposedVars[1]);

    for (Operation &op : oldBody->getOperations()) {
      rewriter.clone(op, mapping);
    }

    rewriter.replaceOp(parallelOp, newParallelOp.getResults());

    return success();
  }

private:
  bool isCollapsible(Value lowerBound, Value upperBound, Value step) const {
    auto lowerConstant = getConstantIntValue(lowerBound);
    auto stepConstant = getConstantIntValue(step);

    return lowerConstant && *lowerConstant == 0 && stepConstant &&
           *stepConstant == 1;
  }
};

class CollapseParallelLoops
    : public impl::CollapseParallelLoopsBase<CollapseParallelLoops> {
public:
  using impl::CollapseParallelLoopsBase<
      CollapseParallelLoops>::CollapseParallelLoopsBase;

  void runOnOperation() final {

    CollapseParallelLoopPattern pattern(&getContext());

    SmallVector<scf::ParallelOp> parallelOps;
    getOperation().walk([&](scf::ParallelOp op) { parallelOps.push_back(op); });

    for (auto parallelOp : parallelOps) {
      PatternRewriter rewriter(&getContext());
      rewriter.setInsertionPoint(parallelOp);
      if (!succeeded(pattern.matchAndRewrite(parallelOp, rewriter))) {
        // Return of matchAndRewrite must be evaluated.
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::transforms
