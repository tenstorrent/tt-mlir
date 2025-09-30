// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICTILECOMPUTELOOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

static SmallVector<int64_t>
calculateOutputSubblockFactors(ArrayRef<int64_t> outputBlockShape,
                               unsigned dstRegisterSizeTiles) {
  // The output operand always corresponds to the compute grid and is therefore
  // the only shape we care about when it comes to constraining on the dst
  // register size. We reverse the output shape to give the priority to the
  // inner dimension and to ensure row major output.
  int64_t remainingBlockFactor = static_cast<int64_t>(dstRegisterSizeTiles);
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

static SmallVector<int64_t> calculateOptimalSubblockSizes(
    ArrayRef<AffineMap> indexingMaps, ValueRange inputs,
    ArrayRef<int64_t> outputBlockShape, unsigned dstRegisterSizeTiles) {
  assert(!indexingMaps.empty());
  MLIRContext *context = indexingMaps[0].getContext();

  SmallVector<int64_t> outputSubblockFactors =
      calculateOutputSubblockFactors(outputBlockShape, dstRegisterSizeTiles);

  //
  // Concat all of the indexing maps together, matmul example:
  // (d0, d1, d2) -> (d0, d2)
  // (d0, d1, d2) -> (d2, d1)
  // (d0, d1, d2) -> (d0, d1)
  // Becomes:
  // (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)
  //
  // We reverse it so that output dimensions get priority for the inverse
  // permutation.
  //
  SmallVector<AffineMap> indexingMapsReversed =
      llvm::to_vector(llvm::reverse(indexingMaps));
  AffineMap concat = concatAffineMaps(indexingMapsReversed, context);

  //
  // Invert the permutation to get a map that we can use to get the buffer
  // factors. Above example becomes:
  // (d0, d1, d2, d3, d4, d5) -> (d0, d3, d1)
  //
  AffineMap inverse = inversePermutation(concat);

  //
  // Since we reversed above to give the output block factors priority in the
  // inverse affine map, we add those first. Then fill the rest of the dims with
  // 1s, these are free variables that don't depend on the sizing of dst. In the
  // future we might do something more intelligent with the free variables, or
  // enable downstream passes like allocation to adjust them based on memory
  // requirements.
  //
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

  auto blockShapes = inverse.compose(flattenedBlockShapes);

  SmallVector<int64_t> subblockSizes(blockShapes.size());
  // Divide block sizes by subblock factors
  for (size_t i = 0; i < blockShapes.size(); i++) {
    subblockSizes[i] = blockShapes[i] / subblockFactors[i];
  }
  return subblockSizes;
}

namespace {
struct D2MGenericComputeRewriter : public OpRewritePattern<linalg::GenericOp> {
  D2MGenericComputeRewriter(MLIRContext *context, unsigned dstRegisterSizeTiles)
      : OpRewritePattern<linalg::GenericOp>(context),
        dstRegisterSizeTiles(dstRegisterSizeTiles) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    assert(op.getOutputs().size() == 1 &&
           "Only one output tensor is supported");
    auto outputTensor =
        mlir::cast<MemRefType>(op.getOutputs().front().getType());
    SmallVector<int64_t> subblockSizes = calculateOptimalSubblockSizes(
        op.getIndexingMapsArray(), op.getInputs(), outputTensor.getShape(),
        dstRegisterSizeTiles);

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

  unsigned dstRegisterSizeTiles;
};
} // namespace

namespace {
class D2MGenericTileComputeLoops
    : public impl::D2MGenericTileComputeLoopsBase<D2MGenericTileComputeLoops> {
public:
  using impl::D2MGenericTileComputeLoopsBase<
      D2MGenericTileComputeLoops>::D2MGenericTileComputeLoopsBase;

  void runOnOperation() final {
    auto device = ttcore::lookupDevice(getOperation());
    assert(device && "Device not found");
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(getOperation());
    auto chipIds = device.getChipIds();
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    unsigned dstRegisterSizeTiles = chipDesc.getDstRegisterSizeTiles();
    if (maxDstRegisterSizeTiles.getValue() > 0) {
      dstRegisterSizeTiles =
          std::min(dstRegisterSizeTiles, maxDstRegisterSizeTiles.getValue());
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericComputeRewriter>(&getContext(),
                                            dstRegisterSizeTiles);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
