// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICTILECOMPUTELOOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

static SmallVector<int64_t>
getSubblockSizes(linalg::GenericOp &op, SmallVector<AffineMap> indexingMaps,
                 SmallVector<int64_t> subblockFactors) {
  // The number of dimensions in the loop space
  unsigned numLoopDims = indexingMaps[0].getNumDims();

  // Create a vector to store the tile sizes for each loop dimension
  SmallVector<int64_t> loopDimTileSizes(numLoopDims, 1);

  // Process each operand (input and output)
  for (unsigned operandIdx = 0; operandIdx < op.getNumOperands();
       ++operandIdx) {
    // Get the indexing map for this operand
    if (operandIdx >= indexingMaps.size()) {
      continue;
    }
    AffineMap map = indexingMaps[operandIdx];

    // Get the corresponding block argument (CB)
    Value cbArg = op.getOperand(operandIdx);

    // Check if the argument is a memref (compute buffer)
    auto memrefType = mlir::dyn_cast<MemRefType>(cbArg.getType());
    if (!memrefType) {
      continue;
    }

    // Get the shape of the memref
    auto cbShape = memrefType.getShape();

    // For each result in the affine map (CB dimension)
    for (unsigned resultIdx = 0; resultIdx < map.getNumResults(); ++resultIdx) {
      // Skip if we're out of bounds for the CB shape
      if (resultIdx >= cbShape.size()) {
        continue;
      }

      // Get the affine expression for this result
      AffineExpr expr = map.getResult(resultIdx);

      // If the expression is a dimension expression, it directly maps to a
      // loop dimension
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        unsigned loopDim = dimExpr.getPosition();

        // If this loop dimension is within our subblock factors range
        if (loopDim < subblockFactors.size()) {
          // Calculate tile size: CB dimension size / subblock factor
          unsigned tileSize = cbShape[resultIdx] / subblockFactors[loopDim];
          // Update the tile size for this loop dimension
          loopDimTileSizes[loopDim] = tileSize;
        }
      }
    }
  }

  return loopDimTileSizes;
}
namespace {
class TTIRGenericComputeRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {

    SmallVector<AffineMap> indexingMaps =
        llvm::map_to_vector(op.getIndexingMaps(), [](Attribute a) {
          return mlir::cast<AffineMapAttr>(a).getValue();
        });
    // Set up tile sizes; fill missing tile sizes at the end with default tile
    // size or tileSize if one was provided.
    SmallVector<int64_t> subblockSizes = getSubblockSizes(
        op, indexingMaps,
        op->getParentOfType<ttir::GenericOp>().getSubblockFactorsValue());

    FailureOr<linalg::TiledLinalgOp> tiledGeneric = linalg::tileLinalgOp(
        rewriter, op,
        linalg::LinalgTilingOptions().setTileSizes(subblockSizes));
    if (failed(tiledGeneric)) {
      return failure();
    }

    rewriter.replaceOp(op, tiledGeneric.value().op);
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericTileComputeLoops
    : public impl::TTIRGenericTileComputeLoopsBase<
          TTIRGenericTileComputeLoops> {
public:
  using impl::TTIRGenericTileComputeLoopsBase<
      TTIRGenericTileComputeLoops>::TTIRGenericTileComputeLoopsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericComputeRewriter>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttir
