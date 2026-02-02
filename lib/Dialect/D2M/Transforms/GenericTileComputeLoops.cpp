// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICTILECOMPUTELOOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

// Determines the subblocking of the output block shape within the constraint of
// the DST capacity.
//
// Example: with output block shape 2x4x8 and 16 tiles in DST, the subblock
// factors will be 2x2x1, which means the output block shape should be further
// partitioned into a 2x2x1 grid of 1x2x8 subblocks, so each subblock fits the
// DST.
static SmallVector<int64_t>
calculateOutputSubblockFactors(ArrayRef<int64_t> outputBlockShape,
                               unsigned dstCapacity) {
  // The output operand always corresponds to the compute grid and is therefore
  // the only shape we care about when it comes to constraining on the dst
  // register size. We reverse the output shape to give the priority to the
  // inner dimension and to ensure row major output.
  int64_t remainingBlockFactor = static_cast<int64_t>(dstCapacity);
  SmallVector<int64_t> outputBlockFactors;
  outputBlockFactors.reserve(outputBlockShape.size());
  for (int64_t dim : llvm::reverse(outputBlockShape)) {
    for (int64_t factor = remainingBlockFactor; factor > 0; factor--) {
      if (dim % factor == 0) {
        outputBlockFactors.push_back(dim / factor);
        // If the dimension is fully consumed by this factor, then we can
        // continue pulling factors from outer dimensions. Otherwise we must
        // snap it to 1 to enforce row major output.
        bool consumed = (dim == factor);
        remainingBlockFactor = consumed ? remainingBlockFactor / factor : 1;
        assert(remainingBlockFactor > 0);
        break;
      }
    }
  }
  assert(outputBlockFactors.size() == outputBlockShape.size());
  // We reversed on the way in, so reverse it back.
  return llvm::to_vector(llvm::reverse(outputBlockFactors));
}

// For a linalg op with multiple operands (inputs & output) that have different
// shapes and indexing patterns, this function determines the blocking of the
// index space of the loops (and thus the subblock sizes of the operands) so
// that the output fits in the DST register.
// Strategy:
// 1. Calculate how to subblock the output based on DST capacity.
// 2. Propagate the constraints back to the loop space using the inverse of the
//    affine map that represents the access patterns of all the operands.
// 3. Dimensions not limited by the DST capacity are free and so do not require
//    subblocking (subblock factor == 1).
// 4. Divide the loop index space by the subblock factors to get the
//    subblocks' index space (the loop blocking) that is consistent across all
//    operands.
static SmallVector<int64_t> calculateOptimalSubblockSizes(
    ArrayRef<AffineMap> indexingMaps, ValueRange inputs,
    ArrayRef<int64_t> outputBlockShape, unsigned dstCapacity) {
  assert(!indexingMaps.empty());

  SmallVector<int64_t> outputSubblockFactors =
      calculateOutputSubblockFactors(outputBlockShape, dstCapacity);

  AffineMap inverse = ttmlir::utils::concatInversePermutationMap(
      llvm::to_vector(indexingMaps), /*reverse=*/true);

  // Since we reversed above to give the output block factors priority in the
  // inverse affine map, we add those first. Then fill the rest of the dims with
  // 1s, these are free variables that don't depend on the sizing of dst. In the
  // future we might do something more intelligent with the free variables, or
  // enable downstream passes like allocation to adjust them based on memory
  // requirements.
  SmallVector<int64_t> flattenedSubblockFactors(outputSubblockFactors);
  flattenedSubblockFactors.resize(inverse.getNumDims(), 1);

  // Eval the affine map to get the subblock factors.
  auto subblockFactors = inverse.compose(flattenedSubblockFactors);

  SmallVector<int64_t> flattenedBlockShapes(outputBlockShape.begin(),
                                            outputBlockShape.end());

  SmallVector<int64_t> inputShapes;
  for (auto input : llvm::reverse(inputs)) {
    auto inputType = mlir::cast<MemRefType>(input.getType());
    inputShapes.append(inputType.getShape().begin(),
                       inputType.getShape().end());
  }

  flattenedBlockShapes.append(inputShapes.begin(), inputShapes.end());

  // From this point on, "block" stands for loop index space.
  auto blockShapes = inverse.compose(flattenedBlockShapes);

  SmallVector<int64_t> subblockSizes(blockShapes.size());
  // Divide block sizes by subblock factors.
  for (size_t i = 0; i < blockShapes.size(); i++) {
    subblockSizes[i] = blockShapes[i] / subblockFactors[i];
  }
  return subblockSizes;
}

namespace {
struct D2MGenericComputeRewriter : public OpRewritePattern<linalg::GenericOp> {
  D2MGenericComputeRewriter(MLIRContext *context,
                            unsigned maxDstPhysicalSizeTiles,
                            bool enableTwoPhaseDestTiling)
      : OpRewritePattern<linalg::GenericOp>(context),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles),
        enableTwoPhaseDestTiling(enableTwoPhaseDestTiling) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Current limita'tion: we only check the output's shape during DST
    // subblocking optimization, ignoring other operands that might reside in
    // the DST and so could lead to overflows.
    //
    // Be conservative: disable loop subblocking if any of the compute op loads
    // more than one DST operand, or isn't in-place w.r.t. DST operands.
    bool optimizeSubblocking = true;
    // op.getRegion().walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
    //   optimizeSubblocking = optimizeSubblocking &&
    //   computeOp.getDstRegInPlace(); optimizeSubblocking =
    //       optimizeSubblocking &&
    //       (computeOp.getOperandsLoadFromDstRegister().size() == 1);
    //   return optimizeSubblocking ? WalkResult::advance()
    //                              : WalkResult::interrupt();
    // });

    TT_assert(op.getRegion().hasOneBlock());
    TT_assertv(op.getOutputs().size() == 1u,
               "Only one output tensor is supported");
    auto outputTensor =
        mlir::cast<MemRefType>(op.getOutputs().front().getType());

    Type largestDstType = utils::getRegionLargestDstElemType(op.getRegion());
    const unsigned baseDstCapacity =
        ttcore::getOpChipDescAttr(op).getDstLogicalSizeTiles(
            largestDstType, false, maxDstPhysicalSizeTiles);
    llvm::errs() << "baseDstCapacity: " << baseDstCapacity << "\n";

    SmallVector<int64_t> subblockSizes(outputTensor.getShape().size(),
                                       enableTwoPhaseDestTiling ? 2 : 1);
    unsigned dstCapacity = baseDstCapacity * (enableTwoPhaseDestTiling ? 2 : 1);
    llvm::errs() << "dstCapacity: " << dstCapacity << "\n";
    if (optimizeSubblocking) {
      subblockSizes = calculateOptimalSubblockSizes(
          op.getIndexingMapsArray(), op.getInputs(), outputTensor.getShape(),
          dstCapacity);
    }

    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setTileSizes(subblockSizes);
    // The use of linalg::LinalgTilingLoopType::AffineLoops was disabled because
    // it caused unexpected behavior where tiling was not applied correctly.
    // Further investigation is needed to determine the root cause before
    // re-enabling it.
    // tilingOptions.setLoopType(linalg::LinalgTilingLoopType::AffineLoops);
    FailureOr<linalg::TiledLinalgOp> tiledGeneric =
        linalg::tileLinalgOp(rewriter, op, tilingOptions);
    if (failed(tiledGeneric)) {
      return failure();
    }

    // Phase 2: If two-phase tiling is enabled, tile the inner linalg.generic
    // again from 2×DST to 1×DST. This creates an inner loop that iterates
    // twice per outer iteration.
    if (enableTwoPhaseDestTiling && optimizeSubblocking) {
      auto innerOp =
          dyn_cast<linalg::GenericOp>(tiledGeneric.value().op.getOperation());
      if (!innerOp) {
        return failure();
      }

      auto innerOutputTensor =
          mlir::cast<MemRefType>(innerOp.getOutputs().front().getType());

      SmallVector<int64_t> phase2SubblockSizes = calculateOptimalSubblockSizes(
          innerOp.getIndexingMapsArray(), innerOp.getInputs(),
          innerOutputTensor.getShape(), baseDstCapacity);

      // Verify that two-phase tiling is valid: the inner output shape must
      // divide evenly by the subblock sizes to produce exactly 2 iterations.
      // TODO(jdesousa): Implement iteration peeling for non-divisible cases.
      int64_t totalIterations = 1;
      ArrayRef<int64_t> innerShape = innerOutputTensor.getShape();
      for (size_t i = 0; i < innerShape.size(); ++i) {
        TT_assertv(
            innerShape[i] % phase2SubblockSizes[i] == 0,
            "Two-phase dest tiling requires output shape to divide evenly by "
            "subblock sizes. Iteration peeling not yet implemented.");
        totalIterations *= innerShape[i] / phase2SubblockSizes[i];
      }
      TT_assertv(totalIterations == 2,
                 "Two-phase dest tiling expects exactly 2 iterations in the "
                 "inner loop (got {0}). Iteration peeling not yet implemented.",
                 totalIterations);

      linalg::LinalgTilingOptions phase2TilingOptions;
      phase2TilingOptions.setTileSizes(phase2SubblockSizes);
      // Note: AffineLoops has known issues where TiledLinalgOp::loops may be
      // empty and the inner linalg shape may not be reduced. Using SCF loops
      // instead ensures correct tiling behavior. The scf.for IVs will later be
      // converted to affine-compatible indices as needed.
      FailureOr<linalg::TiledLinalgOp> phase2TiledGeneric =
          linalg::tileLinalgOp(rewriter, innerOp, phase2TilingOptions);
      if (failed(phase2TiledGeneric)) {
        return failure();
      }

      // Erase the intermediate linalg.generic from phase 1 - it has been
      // replaced by the tiled version from phase 2.
      rewriter.eraseOp(innerOp);

      // Tag the outermost loop from phase 2 tiling with d2m.scratch_space_loop
      // attribute for downstream passes (e.g., fission, scratch allocation).
      // Also convert the SCF loop to an affine loop so its IV can be used with
      // affine.store/load operations.
      if (!phase2TiledGeneric.value().loops.empty()) {
        Operation *outermostScfLoop = phase2TiledGeneric.value().loops.front();
        auto scfForOp = dyn_cast<scf::ForOp>(outermostScfLoop);

        // Check if the scf.for has constant bounds
        std::optional<int64_t> lbOpt, ubOpt, stepOpt;
        if (scfForOp) {
          lbOpt = getConstantIntValue(scfForOp.getLowerBound());
          ubOpt = getConstantIntValue(scfForOp.getUpperBound());
          stepOpt = getConstantIntValue(scfForOp.getStep());
        }

        if (scfForOp && lbOpt && ubOpt && stepOpt) {
          // Convert scf.for to affine.for since it has constant bounds
          int64_t lb = *lbOpt;
          int64_t ub = *ubOpt;
          int64_t step = *stepOpt;

          rewriter.setInsertionPoint(scfForOp);
          auto affineForOp = rewriter.create<affine::AffineForOp>(
              scfForOp.getLoc(), lb, ub, step);

          // Move the body from scf.for to affine.for
          Block *scfBody = scfForOp.getBody();
          Block *affineBody = affineForOp.getBody();

          // Replace uses of scf IV with affine IV
          scfBody->getArgument(0).replaceAllUsesWith(
              affineBody->getArgument(0));

          // Move operations (except terminator) from scf body to affine body
          auto &scfOps = scfBody->getOperations();
          auto &affineOps = affineBody->getOperations();
          affineOps.splice(affineOps.begin(), scfOps, scfOps.begin(),
                           std::prev(scfOps.end()));

          // Set the scratch_space_loop attribute on the new affine.for
          affineForOp->setAttr("d2m.scratch_space_loop",
                               rewriter.getUnitAttr());

          // Erase the old scf.for
          rewriter.eraseOp(scfForOp);
        } else {
          // Fallback: just set the attribute on the SCF loop
          outermostScfLoop->setAttr("d2m.scratch_space_loop",
                                    rewriter.getUnitAttr());
        }
      }

      // Replace the original op with the tiled version from phase 2.
      rewriter.replaceOp(op, phase2TiledGeneric.value().op);
      return success();
    }

    rewriter.replaceOp(op, tiledGeneric.value().op);
    return success();
  }

  unsigned maxDstPhysicalSizeTiles = 0;
  bool enableTwoPhaseDestTiling = false;
};
} // namespace

namespace {
class D2MGenericTileComputeLoops
    : public impl::D2MGenericTileComputeLoopsBase<D2MGenericTileComputeLoops> {
public:
  using impl::D2MGenericTileComputeLoopsBase<
      D2MGenericTileComputeLoops>::D2MGenericTileComputeLoopsBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<D2MGenericComputeRewriter>(
        ctx, maxDstPhysicalSizeTiles.getValue(),
        enableTwoPhaseDestTiling.getValue());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
