// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "d2m-generic-tile-compute-loops"
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

// Convert an scf.for with constant bounds to affine.for in-place.
// Returns the new affine.for, or nullptr if conversion is not possible.
static affine::AffineForOp convertScfForToAffine(scf::ForOp scfForOp,
                                                 PatternRewriter &rewriter) {
  auto lb = getConstantIntValue(scfForOp.getLowerBound());
  auto ub = getConstantIntValue(scfForOp.getUpperBound());
  auto step = getConstantIntValue(scfForOp.getStep());
  if (!lb || !ub || !step) {
    return nullptr;
  }

  rewriter.setInsertionPoint(scfForOp);
  auto affineForOp =
      rewriter.create<affine::AffineForOp>(scfForOp.getLoc(), *lb, *ub, *step);

  Block *scfBody = scfForOp.getBody();
  Block *affineBody = affineForOp.getBody();

  scfBody->getArgument(0).replaceAllUsesWith(affineForOp.getInductionVar());

  auto &scfOps = scfBody->getOperations();
  auto &affineOps = affineBody->getOperations();
  affineOps.splice(affineOps.begin(), scfOps, scfOps.begin(),
                   std::prev(scfOps.end()));

  rewriter.eraseOp(scfForOp);
  return affineForOp;
}

struct D2MGenericComputeRewriter : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const final {
    utils::DstRegisterAnalysis analysis(genericOp);
    const utils::DSTPackingInfo *packingInfo = analysis.lookup(genericOp);
    if (!packingInfo || packingInfo->empty()) {
      return failure();
    }

    SmallVector<linalg::GenericOp> linalgOps;
    genericOp.getRegion(0).walk(
        [&](linalg::GenericOp op) { linalgOps.push_back(op); });

    if (linalgOps.empty()) {
      return failure();
    }

    // Insert unpack_stall_on_pack before any linalg op that reads from an L1
    // buffer written by a preceding linalg op's PACK output. This ensures the
    // UNPACK thread waits for PACK to commit its write before reading.
    llvm::SmallDenseSet<Value> producedValues;
    for (linalg::GenericOp linalgOp : linalgOps) {
      bool needsStall = llvm::any_of(linalgOp.getInputs(), [&](Value v) {
        return producedValues.count(v);
      });
      if (needsStall) {
        rewriter.setInsertionPoint(linalgOp);
        rewriter.create<d2m::UnpackStallOnPackOp>(linalgOp.getLoc());
      }
      for (Value out : linalgOp.getOutputs()) {
        producedValues.insert(out);
      }
    }

    for (linalg::GenericOp linalgOp : linalgOps) {
      Region *parentRegion = linalgOp->getParentRegion();
      const utils::DSTPackingRegionInfo *regionInfo =
          packingInfo->lookup(parentRegion);
      if (!regionInfo) {
        return failure();
      }
      if (failed(tileLinalgOp(linalgOp, *regionInfo, rewriter))) {
        return failure();
      }
    }

    return success();
  }

private:
  LogicalResult tileLinalgOp(linalg::GenericOp linalgOp,
                             const utils::DSTPackingRegionInfo &regionInfo,
                             PatternRewriter &rewriter) const {
    Value outputValue = linalgOp.getOutputs().front();
    auto it = regionInfo.perResult.find(outputValue);
    if (it == regionInfo.perResult.end()) {
      return failure();
    }
    const utils::DSTPackingPerResultInfo &perResult = it->second;

    auto outputType = mlir::cast<MemRefType>(outputValue.getType());

    LLVM_DEBUG({
      llvm::dbgs() << "=== DST Packing for linalg.generic ===\n";
      llvm::dbgs() << "  loc: " << linalgOp.getLoc() << "\n";
      llvm::dbgs() << "  output shape: [";
      llvm::interleaveComma(outputType.getShape(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "  numOuterLoopIters: " << regionInfo.numOuterLoopIters
                   << "\n";
      llvm::dbgs() << "  numTilesPerResult: " << regionInfo.numTilesPerResult
                   << "\n";
      llvm::dbgs() << "  numDstFlips: " << perResult.numDstFlips << "\n";
      llvm::dbgs() << "  numTilesPerFlip: " << perResult.numTilesPerFlip
                   << "\n";
    });

    // Phase 1: Tile to produce numOuterLoopIters outer iterations.
    // Using numTilesPerResult as the effective capacity determines tile
    // sizes that yield the correct number of outer iterations.
    SmallVector<int64_t> phase1TileSizes = calculateOptimalSubblockSizes(
        linalgOp.getIndexingMapsArray(), linalgOp.getInputs(),
        outputType.getShape(),
        static_cast<unsigned>(regionInfo.numTilesPerResult));

    LLVM_DEBUG({
      llvm::dbgs() << "  phase1 tile sizes: [";
      llvm::interleaveComma(phase1TileSizes, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    linalg::LinalgTilingOptions phase1Options;
    phase1Options.setTileSizes(phase1TileSizes);
    FailureOr<linalg::TiledLinalgOp> phase1Tiled =
        linalg::tileLinalgOp(rewriter, linalgOp, phase1Options);
    if (failed(phase1Tiled)) {
      return failure();
    }

    // Phase 2: Tile the inner linalg.generic to produce numDstFlips
    // iterations per outer iteration.
    auto innerOp =
        dyn_cast<linalg::GenericOp>(phase1Tiled.value().op.getOperation());
    if (!innerOp) {
      return failure();
    }

    auto innerOutputType =
        mlir::cast<MemRefType>(innerOp.getOutputs().front().getType());

    SmallVector<int64_t> phase2TileSizes = calculateOptimalSubblockSizes(
        innerOp.getIndexingMapsArray(), innerOp.getInputs(),
        innerOutputType.getShape(),
        static_cast<unsigned>(perResult.numTilesPerFlip));

    LLVM_DEBUG({
      llvm::dbgs() << "  phase1 inner shape: [";
      llvm::interleaveComma(innerOutputType.getShape(), llvm::dbgs());
      llvm::dbgs() << "]\n";
      llvm::dbgs() << "  phase2 tile sizes: [";
      llvm::interleaveComma(phase2TileSizes, llvm::dbgs());
      llvm::dbgs() << "]\n";
    });

    linalg::LinalgTilingOptions phase2Options;
    phase2Options.setTileSizes(phase2TileSizes);
    // Note: linalg::LinalgTilingLoopType::AffineLoops has known issues where
    // TiledLinalgOp::loops may be empty and the inner linalg shape may not
    // be reduced. Using SCF loops and converting to affine immediately after
    // ensures correct tiling behavior with affine-compatible IVs.
    FailureOr<linalg::TiledLinalgOp> phase2Tiled =
        linalg::tileLinalgOp(rewriter, innerOp, phase2Options);
    if (failed(phase2Tiled)) {
      return failure();
    }

    rewriter.eraseOp(innerOp);

    // Convert all phase 2 SCF loops to affine.for so their IVs are usable
    // with affine.store/load in downstream scratch allocation. Tag the
    // outermost loop with d2m.scratch_space_loop for downstream passes.
    if (!phase2Tiled.value().loops.empty()) {
      affine::AffineForOp outermostAffineLoop;

      for (Operation *loopOp : phase2Tiled.value().loops) {
        auto scfForOp = dyn_cast<scf::ForOp>(loopOp);
        if (!scfForOp) {
          continue;
        }

        auto affineForOp = convertScfForToAffine(scfForOp, rewriter);
        if (!affineForOp) {
          continue;
        }

        if (!outermostAffineLoop) {
          outermostAffineLoop = affineForOp;
        }
      }

      if (outermostAffineLoop) {
        outermostAffineLoop->setAttr("d2m.scratch_space_loop",
                                     rewriter.getUnitAttr());
      } else {
        phase2Tiled.value().loops.front()->setAttr("d2m.scratch_space_loop",
                                                   rewriter.getUnitAttr());
      }
    }

    rewriter.eraseOp(linalgOp);
    return success();
  }
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
    patterns.add<D2MGenericComputeRewriter>(ctx);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::d2m
