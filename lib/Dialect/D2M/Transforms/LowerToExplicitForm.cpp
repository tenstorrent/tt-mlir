// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERTOEXPLICITFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Lower a single affine.for with d2m.outer_loop attribute to scf.for.
/// The upper bound operand must already be an arith.constant (i.e.,
/// GetBlockFactorOps should be replaced before calling this).
static void lowerAffineForToSCF(IRRewriter &rewriter,
                                affine::AffineForOp forOp) {
  Location loc = forOp.getLoc();

  // Get the upper bound operand (already a constant after GetBlockFactorOp
  // replacement).
  auto ubOperands = forOp.getUpperBoundOperands();
  assert(ubOperands.size() == 1 && "Expected single upper bound operand");
  Value ub = ubOperands[0];

  // Create lower bound and step constants.
  rewriter.setInsertionPoint(forOp);
  Value lb = arith::ConstantIndexOp::create(rewriter, loc,
                                            forOp.getConstantLowerBound());
  Value step =
      arith::ConstantIndexOp::create(rewriter, loc, forOp.getStepAsInt());

  // Create the scf.for replacement.
  auto scfForOp = scf::ForOp::create(rewriter, loc, lb, ub, step);

  // Preserve the d2m.outer_loop marker attribute.
  scfForOp->setAttr("d2m.outer_loop", rewriter.getUnitAttr());

  Block *affineBody = forOp.getBody();
  Block *scfBody = scfForOp.getBody();

  // Replace all uses of the affine induction variable with the scf one.
  forOp.getInductionVar().replaceAllUsesWith(scfForOp.getInductionVar());

  // Erase the auto-generated scf.yield terminator; we will add our own
  // after splicing.
  rewriter.eraseOp(scfBody->getTerminator());

  // Splice all operations except the affine.yield terminator from the
  // affine body into the scf body.
  scfBody->getOperations().splice(scfBody->end(), affineBody->getOperations(),
                                  affineBody->getOperations().begin(),
                                  std::prev(affineBody->getOperations().end()));

  // Add the scf.yield terminator.
  rewriter.setInsertionPointToEnd(scfBody);
  scf::YieldOp::create(rewriter, loc);

  // Erase the original affine.for (the remaining affine.yield in its body
  // is erased along with it).
  rewriter.eraseOp(forOp);
}

class D2MLowerToExplicitForm
    : public impl::D2MLowerToExplicitFormBase<D2MLowerToExplicitForm> {
public:
  using impl::D2MLowerToExplicitFormBase<
      D2MLowerToExplicitForm>::D2MLowerToExplicitFormBase;

  void runOnOperation() final {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());

    module.walk([&](GenericOp generic) {
      SmallVector<int64_t> blockFactors = generic.getBlockFactorsValue();
      if (blockFactors.empty()) {
        return;
      }

      // Step 1: Replace all GetBlockFactorOps with arith.constant values
      // derived from the GenericOp's block_factors attribute.
      SmallVector<GetBlockFactorOp> blockFactorOps;
      generic.walk([&](GetBlockFactorOp op) { blockFactorOps.push_back(op); });

      for (GetBlockFactorOp op : blockFactorOps) {
        int64_t dim = op.getDim();
        assert(static_cast<size_t>(dim) < blockFactors.size() &&
               "Block factor dimension out of range");
        int64_t value = blockFactors[dim];

        rewriter.setInsertionPoint(op);
        Value constant =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), value);
        rewriter.replaceOp(op, constant);
      }

      // Step 2: Collect affine.for loops marked with d2m.outer_loop and
      // replace them with scf.for loops. The default walk order is
      // post-order (innermost first), which is the correct processing
      // order.
      SmallVector<affine::AffineForOp> outerLoops;
      generic.walk([&](affine::AffineForOp forOp) {
        if (forOp->hasAttr("d2m.outer_loop")) {
          outerLoops.push_back(forOp);
        }
      });

      for (affine::AffineForOp forOp : outerLoops) {
        lowerAffineForToSCF(rewriter, forOp);
      }
    });
  }
};
} // namespace
} // namespace mlir::tt::d2m
