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

#include <cstdint>

namespace mlir::tt::transforms {
#define GEN_PASS_DEF_COLLAPSEPARALLELLOOPS
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

/// Pattern to collapse the last dimensions in SCF parallel loops with 4 or more
/// dimensions and reverse the order of the dimensions.
class CollapseParallelLoopPattern : public OpRewritePattern<scf::ParallelOp> {
private:
  bool isCollapsible(Value lowerBound, Value step) const {
    auto lowerConstant = getConstantIntValue(lowerBound);
    auto stepConstant = getConstantIntValue(step);
    return lowerConstant && *lowerConstant == 0 && stepConstant &&
           *stepConstant == 1;
  }

  scf::ParallelOp createNewParallelOp(PatternRewriter &rewriter,
                                      scf::ParallelOp &oldParallelOp,
                                      uint64_t collapsedSize) const {
    // Last dimensions are collapsed into a single dimension.
    // Because the order of the dimensions is reversed, the first dimension is
    // the collapsed dimensions in the original loop, followed by the second and
    // first dimension.
    Location loc = oldParallelOp.getLoc();
    mlir::Operation::operand_range lowerBounds = oldParallelOp.getLowerBound();
    mlir::Operation::operand_range upperBounds = oldParallelOp.getUpperBound();
    mlir::Operation::operand_range steps = oldParallelOp.getStep();
    mlir::Operation::operand_range initVals = oldParallelOp.getInitVals();

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

    return newParallelOp;
  }

  uint64_t
  calculateProductOfRemainingDims(uint64_t startIdx, uint64_t endIdx,
                                  SmallVector<int64_t> &dimSizes) const {
    uint64_t productOfRemainingDims = 1;
    for (uint64_t i = startIdx; i < endIdx; ++i) {
      productOfRemainingDims *= dimSizes[i - startIdx];
    }
    return productOfRemainingDims;
  }

public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    mlir::Operation::operand_range lowerBounds = parallelOp.getLowerBound();
    mlir::Operation::operand_range upperBounds = parallelOp.getUpperBound();
    mlir::Operation::operand_range steps = parallelOp.getStep();

    if (lowerBounds.size() < 4) {
      return failure();
    }

    uint64_t startIdx = 2;
    uint64_t endIdx = lowerBounds.size();

    for (uint64_t i = startIdx; i < endIdx; ++i) {
      if (!isCollapsible(lowerBounds[i], steps[i])) {
        return failure();
      }
    }

    Location loc = parallelOp.getLoc();
    SmallVector<Value> inductionVars = parallelOp.getInductionVars();

    SmallVector<int64_t> dimSizes;
    int64_t collapsedSize = 1;

    for (uint64_t i = startIdx; i < endIdx; ++i) {
      auto upperConstant = getConstantIntValue(upperBounds[i]);
      auto lowerConstant = getConstantIntValue(lowerBounds[i]);
      if (!upperConstant || !lowerConstant) {
        return failure();
      }
      int64_t dimSize = *upperConstant - *lowerConstant;
      dimSizes.push_back(dimSize);
      collapsedSize *= dimSize;
    }

    scf::ParallelOp newParallelOp =
        createNewParallelOp(rewriter, parallelOp, collapsedSize);

    Block *newBody = newParallelOp.getBody();
    Block *oldBody = parallelOp.getBody();

    newBody->clear();

    rewriter.setInsertionPointToStart(newBody);

    SmallVector<Value> newInductionVars = newParallelOp.getInductionVars();
    IRMapping mapping;
    // Reversing the order of the dimensions.
    // Original loop order: (i1, i2, ..., iN) to (x1, x2, ..., xN)
    // New loop order: (i1, i2, i3) to (x3*x4*...*xN, x2, x1)
    // Mapping induction variables to account for the reversed order and the
    // collapsed dimension.
    mapping.map(inductionVars[0], newInductionVars[2]);
    mapping.map(inductionVars[1], newInductionVars[1]);

    Value collapsedVar = newInductionVars[0];
    SmallVector<Value> decomposedVars;

    Value remaining = collapsedVar;
    // All induction variables need to be derived from the collapsed dimension.
    // Iterator i1 goes from 0 to x3*x4*...*xN.
    // To calculate to what should i3 be mapped to, i1 needs to be divided by
    // x4*x5*...*xN.
    // To ensure that the mapping is correct, the remainder of the division
    // needs to be calculated.
    // The remainder is then used to calculate the next induction variable.
    for (uint64_t i = startIdx; i < endIdx - 1; ++i) {
      int64_t productOfRemainingDims =
          calculateProductOfRemainingDims(i + 1, endIdx, dimSizes);
      Value divisor =
          rewriter.create<arith::ConstantIndexOp>(loc, productOfRemainingDims);
      Value quotient = rewriter.create<arith::DivUIOp>(loc, remaining, divisor);
      decomposedVars.push_back(quotient);

      Value product = rewriter.create<arith::MulIOp>(loc, quotient, divisor);
      remaining = rewriter.create<arith::SubIOp>(loc, remaining, product);
    }
    decomposedVars.push_back(remaining);

    for (uint64_t i = startIdx; i < endIdx; ++i) {
      mapping.map(inductionVars[i], decomposedVars[i - startIdx]);
    }

    for (Operation &op : oldBody->getOperations()) {
      rewriter.clone(op, mapping);
    }

    rewriter.replaceOp(parallelOp, newParallelOp.getResults());

    return success();
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
