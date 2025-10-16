// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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

  AffineMap inverse = utils::concatInversePermutationMap(
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
                            unsigned maxDstPhysicalSizeTiles)
      : OpRewritePattern<linalg::GenericOp>(context),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Current limitation: we only check the output's shape during DST
    // subblocking optimization, ignoring other operands that might reside in
    // the DST and so could lead to overflows.
    //
    // Be conservative: disable loop subblocking if any of the compute op loads
    // more than one DST operand, or isn't in-place w.r.t. DST operands.
    bool optimizeSubblocking = true;
    op.getRegion().walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      optimizeSubblocking = optimizeSubblocking && computeOp.getDstRegInPlace();
      optimizeSubblocking =
          optimizeSubblocking &&
          (computeOp.getOperandsLoadFromDstRegister().size() == 1);
      return optimizeSubblocking ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    assert(op.getRegion().hasOneBlock());
    assert(op.getOutputs().size() == 1 &&
           "Only one output tensor is supported");
    auto outputTensor =
        mlir::cast<MemRefType>(op.getOutputs().front().getType());

    SmallVector<int64_t> subblockSizes(outputTensor.getShape().size(), 1);
    if (optimizeSubblocking) {
      Type largestDstType = utils::getRegionLargestDstElemType(op.getRegion());
      const unsigned dstCapacity =
          ttcore::getOpChipDescAttr(op).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

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

    rewriter.replaceOp(op, tiledGeneric.value().op);
    return success();
  }

  unsigned maxDstPhysicalSizeTiles = 0;
};
} // namespace

namespace {
class D2MGenericTileComputeLoops
    : public impl::D2MGenericTileComputeLoopsBase<D2MGenericTileComputeLoops> {
public:
  using impl::D2MGenericTileComputeLoopsBase<
      D2MGenericTileComputeLoops>::D2MGenericTileComputeLoopsBase;

  void runOnOperation() final {

    // Run the analysis once before any pattern rewriting
    llvm::errs() << "\n=== Running DestRegisterAnalysis BEFORE "
                    "GenericTileComputeLoops ===\n\n";
    DestRegisterAnalysis analysis(getOperation());
    llvm::errs()
        << "\n=== Analysis Complete, Starting Pattern Rewriting ===\n\n";

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<D2MGenericComputeRewriter>(ctx,
                                            maxDstPhysicalSizeTiles.getValue());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
