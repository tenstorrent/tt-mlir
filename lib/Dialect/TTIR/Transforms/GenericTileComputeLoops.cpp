// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICTILECOMPUTELOOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

static SmallVector<unsigned>
getSubblockSizes(Region &computeRegion, SmallVector<AffineMap> indexingMaps,
                 SmallVector<int64_t> subblockFactors) {
  // Get the first block in the compute region
  Block &block = computeRegion.front();

  // The number of dimensions in the loop space
  unsigned numLoopDims = indexingMaps[0].getNumDims();

  // Create a vector to store the tile sizes for each loop dimension
  SmallVector<unsigned> loopDimTileSizes(numLoopDims, 1);

  // Process each operand (input and output)
  for (unsigned operandIdx = 0; operandIdx < block.getNumArguments();
       ++operandIdx) {
    // Get the indexing map for this operand
    if (operandIdx >= indexingMaps.size()) {
      continue;
    }
    AffineMap map = indexingMaps[operandIdx];

    // Get the corresponding block argument (CB)
    BlockArgument cbArg = block.getArgument(operandIdx);

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
class TTIRGenericComputeRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static void
  getTileableBands(Region &computeRegion,
                   std::vector<SmallVector<affine::AffineForOp, 6>> *bands) {
    // Get maximal perfect nest of 'affine.for' insts starting from root
    // (inclusive).
    for (affine::AffineForOp forOp :
         computeRegion.getOps<affine::AffineForOp>()) {
      SmallVector<affine::AffineForOp, 6> band;
      getPerfectlyNestedLoops(band, forOp);
      bands->push_back(band);
    }
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {

    if (!op.isComputeOnlyForm() || op->getRegions().size() != 1) {
      return failure();
    }
    Region &computeRegion = op->getRegion(0);

    std::vector<SmallVector<affine::AffineForOp, 6>> bands;
    getTileableBands(computeRegion, &bands);

    // Tile each band.
    for (auto &band : bands) {
      if (!isTilingValid(band)) {
        band.front().emitRemark("tiling nest is invalid due to dependences");
        continue;
      }

      // Set up tile sizes; fill missing tile sizes at the end with default tile
      // size or tileSize if one was provided.
      SmallVector<unsigned> subblockSizes =
          getSubblockSizes(computeRegion, op.getIndexingMapsValue(),
                           op.getSubblockFactorsValue());
      SmallVector<affine::AffineForOp, 6> tiledNest;
      if (failed(tilePerfectlyNested(band, subblockSizes, &tiledNest))) {
        // An empty band always succeeds.
        assert(!band.empty() && "guaranteed to succeed on empty bands");
        llvm_unreachable("generic compute loop tiling failed!");
      }
    }

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
