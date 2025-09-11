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

/// Pattern to collapse the last two dimensions in 4D SCF parallel loops
class CollapseParallelLoopPattern : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override {
    auto lowerBounds = parallelOp.getLowerBound();
    auto upperBounds = parallelOp.getUpperBound();
    auto steps = parallelOp.getStep();

    // Only handle 4-dimensional parallel loops
    if (lowerBounds.size() != 4) {
      return failure();
    }

    // Check if the last two dimensions are collapsible (zero lower bound, unit
    // step)
    if (!isCollapsible(lowerBounds[2], upperBounds[2], steps[2]) ||
        !isCollapsible(lowerBounds[3], upperBounds[3], steps[3])) {
      return failure();
    }

    // Collapse dimensions 2 and 3 (the last two dimensions)
    unsigned startIdx = 2;
    unsigned endIdx = 3;
    Location loc = parallelOp.getLoc();
    auto initVals = parallelOp.getInitVals();
    auto inductionVars = parallelOp.getInductionVars();

    // Calculate the collapsed dimension size
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

    // Build new bounds, steps, and induction variables in REVERSED order
    SmallVector<Value> newLowerBounds, newUpperBounds, newSteps;

    // Add the collapsed dimension FIRST (was last)
    newLowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    newUpperBounds.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, collapsedSize));
    newSteps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));

    // Add dimension 1 (middle dimension)
    newLowerBounds.push_back(lowerBounds[1]);
    newUpperBounds.push_back(upperBounds[1]);
    newSteps.push_back(steps[1]);

    // Add dimension 0 (first dimension) LAST
    newLowerBounds.push_back(lowerBounds[0]);
    newUpperBounds.push_back(upperBounds[0]);
    newSteps.push_back(steps[0]);

    // Create the new parallel loop
    auto newParallelOp = rewriter.create<scf::ParallelOp>(
        loc, newLowerBounds, newUpperBounds, newSteps, initVals);

    // Clone the entire body region to the new parallel loop
    Block *newBody = newParallelOp.getBody();
    Block *oldBody = parallelOp.getBody();

    // Clear the auto-generated terminator in the new body
    newBody->clear();

    // Set insertion point to the new body
    rewriter.setInsertionPointToStart(newBody);

    // Create mappings for the new induction variables (in reversed order)
    SmallVector<Value> newInductionVars = newParallelOp.getInductionVars();
    IRMapping mapping;

    // New order: [merged, dim1, dim0] maps to original [dim0, dim1, dim2, dim3]
    // newInductionVars[0] = merged (will be decomposed to dim2, dim3)
    // newInductionVars[1] = original dim1
    // newInductionVars[2] = original dim0

    // Map original dim0 to newInductionVars[2] (last position)
    mapping.map(inductionVars[0], newInductionVars[2]);

    // Map original dim1 to newInductionVars[1] (middle position)
    mapping.map(inductionVars[1], newInductionVars[1]);

    // Decompose the collapsed induction variable (newInductionVars[0])
    Value collapsedVar = newInductionVars[0];
    SmallVector<Value> decomposedVars;

    Value remaining = collapsedVar;
    for (unsigned i = startIdx; i <= endIdx; ++i) {
      if (i == endIdx) {
        // Last dimension - use the remainder
        decomposedVars.push_back(remaining);
      } else {
        // Decompose: var_i = remaining / (product of remaining dims)
        int64_t productOfRemainingDims = 1;
        for (unsigned j = i + 1; j <= endIdx; ++j) {
          productOfRemainingDims *= dimSizes[j - startIdx];
        }

        Value divisor = rewriter.create<arith::ConstantIndexOp>(
            loc, productOfRemainingDims);
        Value quotient =
            rewriter.create<arith::DivUIOp>(loc, remaining, divisor);
        decomposedVars.push_back(quotient);

        // Update remaining for next iteration
        Value product = rewriter.create<arith::MulIOp>(loc, quotient, divisor);
        remaining = rewriter.create<arith::SubIOp>(loc, remaining, product);
      }
    }

    // Map the decomposed variables (original dim2 and dim3)
    mapping.map(inductionVars[2], decomposedVars[0]); // dim2 (k)
    mapping.map(inductionVars[3], decomposedVars[1]); // dim3 (l)

    // Clone all operations from the old body with the new mappings
    for (Operation &op : oldBody->getOperations()) {
      rewriter.clone(op, mapping);
    }

    // Replace the old parallel loop
    rewriter.replaceOp(parallelOp, newParallelOp.getResults());

    return success();
  }

private:
  /// Check if a dimension is collapsible (zero lower bound, unit step)
  bool isCollapsible(Value lowerBound, Value upperBound, Value step) const {
    auto lowerConstant = getConstantIntValue(lowerBound);
    auto stepConstant = getConstantIntValue(step);

    return lowerConstant && *lowerConstant == 0 && stepConstant &&
           *stepConstant == 1;
  }
};

/// Pass implementation
class CollapseParallelLoops
    : public impl::CollapseParallelLoopsBase<CollapseParallelLoops> {
public:
  using impl::CollapseParallelLoopsBase<
      CollapseParallelLoops>::CollapseParallelLoopsBase;

  void runOnOperation() final {
    // Debug: Count scf.parallel operations before pattern application
    unsigned parallelOpCount = 0;
    getOperation().walk([&](scf::ParallelOp op) { parallelOpCount++; });

    // Apply patterns manually to avoid canonicalization
    CollapseParallelLoopPattern pattern(&getContext());

    SmallVector<scf::ParallelOp> parallelOps;
    getOperation().walk([&](scf::ParallelOp op) { parallelOps.push_back(op); });

    for (auto parallelOp : parallelOps) {
      PatternRewriter rewriter(&getContext());
      rewriter.setInsertionPoint(parallelOp);
      if (!succeeded(pattern.matchAndRewrite(parallelOp, rewriter))) {
        llvm::errs() << "Failed to apply pattern to parallel op\n";
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::transforms
