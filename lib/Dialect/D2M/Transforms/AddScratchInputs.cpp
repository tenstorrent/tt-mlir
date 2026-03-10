// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "D2MAddScratchInputs"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MADDSCRATCHINPUTS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Fixed scratch buffer size in bytes.
constexpr size_t kScratchSizeBytes = 128 * 1024; // 128KB

// Get the tile type from a memref type, if it has one.
static ttcore::TileType getTileType(MemRefType memrefType) {
  return mlir::dyn_cast<ttcore::TileType>(memrefType.getElementType());
}

// Find a tiled input operand and return its memref type.
// Returns a null MemRefType if no tiled input is found.
static MemRefType findTiledInputType(GenericOp genericOp) {
  for (Value input : genericOp.getInputs()) {
    auto memrefType = mlir::dyn_cast<MemRefType>(input.getType());
    if (!memrefType) {
      continue;
    }
    if (getTileType(memrefType)) {
      return memrefType;
    }
  }
  return MemRefType();
}

// Check if a d2m.generic needs a scratch buffer. Scratch is added when the
// compute region contains more than one binary FPU op (add/sub/mul), which
// indicates a fused kernel that may need to spill intermediate results.
// TODO(ckaravasilisTT): Use a more precise method to determine if a generic
// needs a scratch buffer and of what size.
static bool needsScratch(GenericOp genericOp) {
  if (genericOp.getNumRegions() == 0) {
    return false;
  }

  unsigned binaryFPUCount = 0;
  genericOp.getRegion(0).walk([&](Operation *op) {
    if (isa<TileAddOp, TileSubOp, TileMulOp>(op)) {
      ++binaryFPUCount;
    }
  });

  return binaryFPUCount > 1;
}

// Add scratch input to a single d2m.generic op (post-bufferization).
// Creates a memref.alloc for the scratch buffer and rebuilds the generic with
// the scratch as an additional input.
static LogicalResult addScratchToGeneric(GenericOp genericOp) {
  // Skip if this generic already has scratch inputs.
  if (genericOp.getScratchInputsAttr()) {
    return failure();
  }

  // Skip if not in compute-only form.
  if (!genericOp.isComputeOnlyForm()) {
    return failure();
  }

  // Only add scratch to fused generics with multiple binary FPU ops
  // (add/sub/mul) that may need to spill intermediate results.
  if (!needsScratch(genericOp)) {
    return failure();
  }

  // Find a tiled input to get the tile type and grid shape.
  MemRefType refMemRefType = findTiledInputType(genericOp);
  if (!refMemRefType) {
    return failure();
  }

  auto refDeviceLayout =
      ttcore::getDeviceLayout(mlir::cast<ShapedType>(refMemRefType));
  if (!refDeviceLayout) {
    return failure();
  }

  ttcore::TileType tileType = getTileType(refMemRefType);
  if (!tileType) {
    return failure();
  }

  // Calculate number of tiles that fit in the scratch buffer.
  size_t tileSizeBytes = tileType.getSizeBytes();
  size_t numTiles = kScratchSizeBytes / tileSizeBytes;
  if (numTiles == 0) {
    numTiles = 1; // At least one tile.
  }

  // Get grid shape from reference input.
  auto gridShape =
      refDeviceLayout.getGridShape(mlir::cast<ShapedType>(refMemRefType));

  // Build scratch memref shape: [grid_dims..., 1, numTiles].
  SmallVector<int64_t> scratchShape;
  for (auto dim : gridShape) {
    scratchShape.push_back(dim);
  }
  scratchShape.push_back(1);
  scratchShape.push_back(static_cast<int64_t>(numTiles));

  // Build shard shape: [1, numTiles].
  SmallVector<int64_t> scratchShardShape = {1, static_cast<int64_t>(numTiles)};

  // Create ShardLayoutAttr from shard shape and tile type.
  auto scratchLayout =
      ttcore::ShardLayoutAttr::get(scratchShardShape, tileType, /*buffers=*/1);

  auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
      genericOp.getContext(), ttcore::MemorySpace::DeviceL1);

  // Full device memref type for the alloc (grid + shard dims).
  MemRefType scratchMemRefType =
      MemRefType::get(scratchShape, tileType, scratchLayout, l1MemorySpace);

  // Shard memref type for the CB block arg.
  MemRefType scratchShardMemRefType = MemRefType::get(
      scratchShardShape, tileType, /*layout=*/nullptr, l1MemorySpace);

  // Create memref.alloc for the scratch buffer before the generic.
  OpBuilder builder(genericOp);
  auto scratchAlloc =
      builder.create<memref::AllocOp>(genericOp.getLoc(), scratchMemRefType);

  // Build new inputs with scratch added at the end.
  unsigned numOldInputs = genericOp.getInputs().size();
  int64_t scratchInputIndex = static_cast<int64_t>(numOldInputs);

  SmallVector<Value> newInputs(genericOp.getInputs());
  newInputs.push_back(scratchAlloc.getResult());

  // Update indexing maps: insert scratch's map before output maps.
  // Scratch uses a constant map (all zeros).
  SmallVector<Attribute> newIndexingMaps;
  auto oldMaps = genericOp.getIndexingMaps();
  for (unsigned i = 0; i < numOldInputs; ++i) {
    newIndexingMaps.push_back(oldMaps[i]);
  }

  // Add constant map for scratch.
  unsigned numDims = genericOp.getNumDims();
  SmallVector<AffineExpr> zeroExprs(numDims, builder.getAffineConstantExpr(0));
  AffineMap scratchMap =
      AffineMap::get(numDims, 0, zeroExprs, builder.getContext());
  newIndexingMaps.push_back(AffineMapAttr::get(scratchMap));

  // Add output maps.
  for (unsigned i = numOldInputs; i < oldMaps.size(); ++i) {
    newIndexingMaps.push_back(oldMaps[i]);
  }

  // Create scratch_inputs attribute.
  auto scratchInputsAttr = builder.getDenseI64ArrayAttr({scratchInputIndex});

  // Create the new GenericOp with empty regions.
  auto newGenericOp = builder.create<GenericOp>(
      genericOp.getLoc(), genericOp.getResultTypes(), newInputs,
      genericOp.getOutputs(), genericOp.getAdditionalArgs(),
      genericOp.getGrid(), genericOp.getBlockFactors(),
      builder.getArrayAttr(newIndexingMaps), genericOp.getIteratorTypes(),
      genericOp.getThreads(), scratchInputsAttr,
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

    // Add scratch block arg type: CB wrapping the shard memref.
    Type scratchBlockArgType = CBType::get(scratchShardMemRefType);
    newBlockArgTypes.push_back(scratchBlockArgType);
    newBlockArgLocs.push_back(genericOp.getLoc());

    // Copy old output block arg types.
    for (unsigned i = numOldInputs; i < oldBlock->getNumArguments(); ++i) {
      newBlockArgTypes.push_back(oldBlock->getArgument(i).getType());
      newBlockArgLocs.push_back(oldBlock->getArgument(i).getLoc());
    }

    Block *newBlock = builder.createBlock(&newRegion, newRegion.end(),
                                          newBlockArgTypes, newBlockArgLocs);

    builder.setInsertionPointToStart(newBlock);

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

    // Clone operations after the get_scratch_from_cb.
    for (Operation &op : oldBlock->without_terminator()) {
      builder.clone(op, mapping);
    }

    // Clone terminator if present.
    if (oldBlock->mightHaveTerminator()) {
      builder.clone(*oldBlock->getTerminator(), mapping);
    }
  }

  genericOp.erase();
  return success();
}

class D2MAddScratchInputs
    : public impl::D2MAddScratchInputsBase<D2MAddScratchInputs> {
  using D2MAddScratchInputsBase::D2MAddScratchInputsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Collect generics.
    SmallVector<GenericOp> genericsToProcess;
    moduleOp.walk(
        [&](GenericOp genericOp) { genericsToProcess.push_back(genericOp); });

    for (GenericOp genericOp : genericsToProcess) {
      // addScratchToGeneric returns failure for generics that don't need
      // scratch (not an error). We intentionally ignore the result.
      (void)addScratchToGeneric(genericOp);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
