// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "D2MAddScratchInputs"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MADDSCRATCHINPUTS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Fixed scratch buffer size in bytes.
constexpr size_t kScratchSizeBytes = 128 * 1024; // 128KB

/// Get the tile type from a tensor type, if it has one.
static ttcore::TileType getTileType(RankedTensorType tensorType) {
  return mlir::dyn_cast<ttcore::TileType>(tensorType.getElementType());
}

/// Find a tiled input operand and return its tensor type.
/// Returns nullptr if no tiled input is found.
static RankedTensorType findTiledInputType(GenericOp genericOp) {
  for (Value input : genericOp.getInputs()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType) {
      continue;
    }
    if (getTileType(tensorType)) {
      return tensorType;
    }
  }
  return nullptr;
}

/// Check if a d2m.generic is a fused kernel by counting linalg.generic ops
/// inside its compute region. A fused kernel has more than one linalg.generic.
static bool isFusedGeneric(GenericOp genericOp) {
  if (genericOp.getNumRegions() == 0) {
    return false;
  }

  unsigned linalgGenericCount = 0;
  genericOp.getRegion(0).walk([&](linalg::GenericOp) { ++linalgGenericCount; });

  return linalgGenericCount > 1;
}

/// Pattern to add a scratch input to fused d2m.generic ops.
/// Scratch is a fixed 128KB buffer for FPU fusion temporary storage.
struct AddScratchInputPattern : public OpRewritePattern<GenericOp> {
  explicit AddScratchInputPattern(MLIRContext *context)
      : OpRewritePattern<GenericOp>(context) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const final {
    // Only process ops in tensor space (pre-bufferization).
    if (!genericOp.hasPureTensorSemantics()) {
      return failure();
    }

    // Skip if this generic already has scratch inputs.
    if (genericOp.getScratchInputsAttr()) {
      return failure();
    }

    // Skip if not in compute-only form (before datamovement is generated).
    if (!genericOp.isComputeOnlyForm()) {
      return failure();
    }

    // Only add scratch to FUSED generics (those with multiple linalg.generic
    // ops from elementwise fusion). Non-fused ops don't need scratch.
    if (!isFusedGeneric(genericOp)) {
      return failure();
    }

    // Find a tiled input to get the tile type.
    RankedTensorType refTensorType = findTiledInputType(genericOp);
    if (!refTensorType) {
      return failure();
    }

    auto refLayout = mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(
        refTensorType.getEncoding());
    if (!refLayout) {
      return failure();
    }

    ttcore::TileType tileType = getTileType(refTensorType);
    if (!tileType) {
      return failure();
    }

    // Calculate number of tiles that fit in the scratch buffer.
    size_t tileSizeBytes = tileType.getSizeBytes();
    size_t numTiles = kScratchSizeBytes / tileSizeBytes;
    if (numTiles == 0) {
      numTiles = 1; // At least one tile.
    }

    // Build scratch tensor shape: [grid_dims..., 1, numTiles].
    // The logical shape needs to match grid structure. We copy the grid
    // dimensions from the reference and set shard dimensions to accommodate
    // our tile count.
    auto gridShape = refLayout.getGridShape(refTensorType);

    SmallVector<int64_t> scratchShape;
    for (auto dim : gridShape) {
      scratchShape.push_back(dim);
    }
    scratchShape.push_back(1);
    scratchShape.push_back(static_cast<int64_t>(numTiles));

    // Create MetalLayoutAttr for scratch.
    auto scratchLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), scratchShape, ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceL1, ttcore::TensorMemoryLayout::Sharded);

    RankedTensorType scratchTensorType =
        RankedTensorType::get(scratchShape, tileType, scratchLayout);

    // Create the d2m.empty op for the scratch tensor.
    rewriter.setInsertionPoint(genericOp);
    auto scratchEmpty =
        rewriter.create<EmptyOp>(genericOp.getLoc(), scratchTensorType);

    // Build new inputs with scratch added at the end.
    unsigned numOldInputs = genericOp.getInputs().size();
    int64_t scratchInputIndex = static_cast<int64_t>(numOldInputs);

    SmallVector<Value> newInputs(genericOp.getInputs());
    newInputs.push_back(scratchEmpty.getResult());

    // Update indexing maps: insert scratch's map before output maps.
    // Scratch uses a CONSTANT map (all zeros) - it broadcasts to all tiles.
    SmallVector<Attribute> newIndexingMaps;
    auto oldMaps = genericOp.getIndexingMaps();
    for (unsigned i = 0; i < numOldInputs; ++i) {
      newIndexingMaps.push_back(oldMaps[i]);
    }

    // Add constant map for scratch (broadcasts to all loop iterations).
    unsigned numDims = genericOp.getNumDims();
    SmallVector<AffineExpr> zeroExprs(numDims,
                                      rewriter.getAffineConstantExpr(0));
    AffineMap scratchMap =
        AffineMap::get(numDims, 0, zeroExprs, rewriter.getContext());
    newIndexingMaps.push_back(AffineMapAttr::get(scratchMap));

    // Add output maps.
    for (unsigned i = numOldInputs; i < oldMaps.size(); ++i) {
      newIndexingMaps.push_back(oldMaps[i]);
    }

    // Create scratch_inputs attribute.
    auto scratchInputsAttr = rewriter.getDenseI64ArrayAttr({scratchInputIndex});

    // Create the new GenericOp with empty regions.
    auto newGenericOp = rewriter.create<GenericOp>(
        genericOp.getLoc(), genericOp.getResultTypes(), newInputs,
        genericOp.getOutputs(), genericOp.getGrid(),
        genericOp.getBlockFactors(), rewriter.getArrayAttr(newIndexingMaps),
        genericOp.getIteratorTypes(), genericOp.getThreads(), scratchInputsAttr,
        /*numRegions=*/genericOp.getNumRegions());

    // Clone regions from old op to new op.
    for (auto [oldRegion, newRegion] :
         llvm::zip(genericOp.getRegions(), newGenericOp.getRegions())) {
      if (oldRegion.empty()) {
        continue;
      }

      Block *oldBlock = &oldRegion.front();

      // Build new block argument types.
      // Block args match operands: [input0, ..., inputN, scratch, output0, ...]
      SmallVector<Type> newBlockArgTypes;
      SmallVector<Location> newBlockArgLocs;

      // Copy old input block arg types.
      for (unsigned i = 0; i < numOldInputs; ++i) {
        newBlockArgTypes.push_back(oldBlock->getArgument(i).getType());
        newBlockArgLocs.push_back(oldBlock->getArgument(i).getLoc());
      }

      // Add scratch block arg type with shard shape [1, numTiles].
      SmallVector<int64_t> scratchShardShape = {1,
                                                static_cast<int64_t>(numTiles)};
      Type scratchBlockArgType =
          CBType::get(RankedTensorType::get(scratchShardShape, tileType));
      newBlockArgTypes.push_back(scratchBlockArgType);
      newBlockArgLocs.push_back(genericOp.getLoc());

      // Copy old output block arg types.
      for (unsigned i = numOldInputs; i < oldBlock->getNumArguments(); ++i) {
        newBlockArgTypes.push_back(oldBlock->getArgument(i).getType());
        newBlockArgLocs.push_back(oldBlock->getArgument(i).getLoc());
      }

      Block *newBlock = rewriter.createBlock(&newRegion, newRegion.end(),
                                             newBlockArgTypes, newBlockArgLocs);

      // Build mapping from old block args to new block args.
      IRMapping mapping;

      // Map old input args (indices 0 to numOldInputs-1) -> same indices.
      for (unsigned i = 0; i < numOldInputs; ++i) {
        mapping.map(oldBlock->getArgument(i), newBlock->getArgument(i));
      }

      // Scratch is at index numOldInputs in new block (unmapped, no old arg).

      // Map old output args -> shifted by 1 (because scratch is inserted).
      for (unsigned i = numOldInputs; i < oldBlock->getNumArguments(); ++i) {
        mapping.map(oldBlock->getArgument(i), newBlock->getArgument(i + 1));
      }

      // Clone operations.
      rewriter.setInsertionPointToStart(newBlock);
      for (Operation &op : oldBlock->without_terminator()) {
        rewriter.clone(op, mapping);
      }

      // Clone terminator if present.
      if (oldBlock->mightHaveTerminator()) {
        rewriter.clone(*oldBlock->getTerminator(), mapping);
      }
    }

    rewriter.replaceOp(genericOp, newGenericOp.getResults());
    return success();
  }
};

class D2MAddScratchInputs
    : public impl::D2MAddScratchInputsBase<D2MAddScratchInputs> {
  using D2MAddScratchInputsBase::D2MAddScratchInputsBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AddScratchInputPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
