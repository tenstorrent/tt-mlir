// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERTOEXPLICITFORM
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Lower a single affine.for with d2m.blocking_loop attribute to scf.for.
/// The upper bound operand must already be an arith.constant (i.e.,
/// GetBlockFactorOps should be replaced before calling this).
static void lowerBlockingAffineLoopsToSCF(IRRewriter &rewriter,
                                          affine::AffineForOp forOp) {
  Location loc = forOp.getLoc();

  // Get the upper bound operand (already a constant after GetBlockFactorOp
  // replacement).
  auto ubOperands = forOp.getUpperBoundOperands();
  TT_assertv(ubOperands.size() == 1u,
             "Expected single upper bound operand for d2m.blocking_loop");
  Value ub = ubOperands[0];

  // Create lower bound and step constants.
  rewriter.setInsertionPoint(forOp);
  Value lb = arith::ConstantIndexOp::create(rewriter, loc,
                                            forOp.getConstantLowerBound());
  Value step =
      arith::ConstantIndexOp::create(rewriter, loc, forOp.getStepAsInt());

  // Create the scf.for replacement.
  auto scfForOp = scf::ForOp::create(rewriter, loc, lb, ub, step);

  // Preserve the d2m.blocking_loop marker attribute (carries block factor
  // index).
  scfForOp->setAttr(utils::kBlockingLoopAttr,
                    forOp->getAttr(utils::kBlockingLoopAttr));

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

/// Build a lookup table from loop dimension -> grid dimension index.
/// A loop dimension participates in grid offsets when that dim appears in one
/// of the output indexing map's grid result positions.
static SmallVector<int64_t> buildLoopDimToGridDimMap(GenericOp generic,
                                                     size_t numLoopDims) {
  SmallVector<int64_t> loopDimToGridDim(numLoopDims, -1);

  unsigned outputOperandIndex = generic.getOutputs().getBeginOperandIndex();
  AffineMap outputOperandIndexingMap =
      generic.getIndexingMap(outputOperandIndex);

  // The first output map results correspond to grid dimensions. The grid
  // mapping includes a leading device id result when present.
  AffineMap gridMapping = generic.getGrid().getMapping();
  constexpr unsigned numPhysicalGridDims = 2;
  unsigned numGridDims = gridMapping.isEmpty()
                             ? numPhysicalGridDims
                             : gridMapping.getNumResults() - 1;

  for (size_t loopDim = 0; loopDim < numLoopDims; ++loopDim) {
    AffineExpr dimExpr = mlir::getAffineDimExpr(
        static_cast<unsigned>(loopDim), outputOperandIndexingMap.getContext());
    std::optional<unsigned> resultPos =
        outputOperandIndexingMap.getResultPosition(dimExpr);
    if (resultPos && *resultPos < numGridDims) {
      loopDimToGridDim[loopDim] = static_cast<int64_t>(*resultPos);
    }
  }

  return loopDimToGridDim;
}

/// Lower d2m.block_offset(dim) into block_factor(dim) * d2m.core_index(gridDim)
/// when the loop dim participates in the grid. Otherwise lower to 0.
static void lowerBlockOffsetOps(IRRewriter &rewriter, GenericOp generic,
                                ArrayRef<int64_t> blockFactors) {
  SmallVector<int64_t> loopDimToGridDim =
      buildLoopDimToGridDimMap(generic, blockFactors.size());
  AffineMap gridMapping = generic.getGrid().getMapping();

  SmallVector<BlockOffsetOp> blockOffsetOps;
  generic.walk([&](BlockOffsetOp op) { blockOffsetOps.push_back(op); });

  for (BlockOffsetOp op : blockOffsetOps) {
    int64_t dim = op.getDim();
    TT_assertv(static_cast<size_t>(dim) < blockFactors.size(),
               "Block offset dimension {} out of range for block_factors size "
               "{}",
               dim, blockFactors.size());

    rewriter.setInsertionPoint(op);
    int64_t gridDim = loopDimToGridDim[dim];
    if (gridDim < 0) {
      Value zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
      rewriter.replaceOp(op, zero);
      continue;
    }

    Value blockFactorConstant = arith::ConstantIndexOp::create(
        rewriter, op.getLoc(), blockFactors[dim]);
    Value coreIndex =
        rewriter.create<CoreIndexOp>(op.getLoc(), gridDim, gridMapping);
    Value blockOffset = rewriter.create<arith::MulIOp>(
        op.getLoc(), blockFactorConstant, coreIndex);
    rewriter.replaceOp(op, blockOffset);
  }
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
        TT_assertv(static_cast<size_t>(dim) < blockFactors.size(),
                   "Block factor dimension {} out of range for block_factors "
                   "size {}",
                   dim, blockFactors.size());
        int64_t value = blockFactors[dim];

        rewriter.setInsertionPoint(op);
        Value constant =
            arith::ConstantIndexOp::create(rewriter, op.getLoc(), value);
        rewriter.replaceOp(op, constant);
      }

      // Step 2: Replace all BlockOffsetOps with
      // block_factor_constant * core_index.
      lowerBlockOffsetOps(rewriter, generic, blockFactors);

      // Step 3: Collect affine.for loops marked with d2m.blocking_loop and
      // replace them with scf.for loops. The default walk order is
      // post-order (innermost first), which is the correct processing
      // order.
      SmallVector<affine::AffineForOp> outerLoops;
      generic.walk([&](affine::AffineForOp forOp) {
        if (forOp->hasAttr(utils::kBlockingLoopAttr)) {
          outerLoops.push_back(forOp);
        }
      });

      for (affine::AffineForOp forOp : outerLoops) {
        lowerBlockingAffineLoopsToSCF(rewriter, forOp);
      }
    });

    // Step 4: Convert all generic ops to explicit datamovement form by
    // clearing symbolic indexing/blocking metadata once lowering is complete.
    module.walk([&](GenericOp generic) {
      rewriter.modifyOpInPlace(generic, [&]() {
        generic.setBlockFactorsAttr(rewriter.getI64ArrayAttr({}));
        generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr({}));
        generic.setIteratorTypesAttr(rewriter.getArrayAttr({}));
      });
    });

    // Force lowering of affine apply/map arithmetic generated by explicit form
    // rewrites into arith ops.
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::d2m
