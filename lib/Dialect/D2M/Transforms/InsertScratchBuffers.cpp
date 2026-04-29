// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "D2MInsertScratchBuffers"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTSCRATCHBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Fixed scratch buffer size in bytes.
constexpr size_t kScratchSizeBytes = 128 * 1024; // 128KB

// Get the tile type from a memref type, if it has one.
static ttcore::TileType getTileType(MemRefType memrefType) {
  return mlir::dyn_cast<ttcore::TileType>(memrefType.getElementType());
}

// Find a tiled memref operand that can serve as the scratch type reference.
// Prefer tiled inputs when present, but fall back to tiled outputs.
// TO-DO (ckaravasilis): A better semantic criterion based on intermediate
// analysis rather than relying on input/output.
static MemRefType findTiledReferenceType(GenericOp genericOp) {
  auto findTiledType = [](ValueRange values) -> MemRefType {
    for (Value value : values) {
      auto memrefType = mlir::dyn_cast<MemRefType>(value.getType());
      if (!memrefType) {
        continue;
      }
      if (getTileType(memrefType)) {
        return memrefType;
      }
    }
    return MemRefType();
  };

  if (MemRefType tiledInputType = findTiledType(genericOp.getInputs())) {
    return tiledInputType;
  }
  return findTiledType(genericOp.getOutputs());
}

// Check if a d2m.generic needs a scratch buffer. Scratch is needed when the
// region contains more than one linalg.generic, which means elementwise fusion
// produced a multi-op kernel whose intermediate results must be spilled to L1.
static bool needsScratch(GenericOp genericOp) {
  if (genericOp.getNumRegions() == 0) {
    return false;
  }

  unsigned linalgCount = 0;
  genericOp.getRegion(0).walk([&](linalg::GenericOp) { ++linalgCount; });
  return linalgCount > 1;
}

// Add a scratch buffer inside a single d2m.generic op's region
// (post-bufferization). Creates a memref.alloc + scratch_init at the start of
// the region body.
static void addScratchToGeneric(GenericOp genericOp) {
  // Skip if not in compute-only form.
  if (!genericOp.isComputeOnlyForm()) {
    return;
  }

  // Only add scratch to fused generics with multiple linalg.generic ops,
  // which indicates elementwise fusion produced a multi-op kernel whose
  // intermediate results must be spilled to L1.
  if (!needsScratch(genericOp)) {
    return;
  }

  MemRefType refMemRefType = findTiledReferenceType(genericOp);
  if (!refMemRefType) {
    return;
  }

  ttcore::TileType tileType = getTileType(refMemRefType);

  // Calculate number of tiles that fit in the scratch buffer.
  size_t tileSizeBytes = tileType.getSizeBytes();
  size_t numTiles = kScratchSizeBytes / tileSizeBytes;
  if (numTiles == 0) {
    numTiles = 1; // At least one tile.
  }

  // Build scratch shard shape: [1, numTiles].
  SmallVector<int64_t> scratchShardShape = {1, static_cast<int64_t>(numTiles)};

  // Create CBLayoutAttr for the scratch buffer (single-buffered).
  auto cbLayout =
      ttcore::CBLayoutAttr::get(scratchShardShape, tileType, /*buffers=*/1);

  auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
      genericOp.getContext(), ttcore::MemorySpace::DeviceL1);

  // Shard memref type for the scratch buffer inside the region.
  MemRefType scratchShardMemRefType =
      MemRefType::get(scratchShardShape, tileType, cbLayout, l1MemorySpace);

  // Insert memref.alloc + scratch_init at the start of the region body.
  Block &block = genericOp.getRegion(0).front();
  OpBuilder builder(&block, block.begin());

  auto scratchAlloc = builder.create<memref::AllocOp>(genericOp.getLoc(),
                                                      scratchShardMemRefType);
  builder.create<ScratchInitOp>(genericOp.getLoc(), scratchAlloc.getResult());
}

// Transfer d2m.blocking_map attributes from inner linalg ops onto the
// memref.alloc backing each linalg op's output (DPS init). This is a
// post-bufferization step: ElementwiseFusion sets d2m.blocking_map on each
// linalg op (which survives bufferization), but the actual reblocking in
// the Allocator inspects the attribute on the memref.alloc. The alloc that
// corresponds to a linalg's DPS init is exactly the intermediate buffer
// produced by elementwise fusion.
static void transferBlockingMaps(GenericOp genericOp) {
  if (genericOp.getNumRegions() == 0) {
    return;
  }
  for (Operation &op : genericOp.getRegion(0).front()) {
    auto linalgOp = dyn_cast<linalg::GenericOp>(&op);
    if (!linalgOp) {
      continue;
    }
    auto blockingMap = linalgOp->getAttr("d2m.blocking_map");
    if (!blockingMap) {
      continue;
    }
    for (OpOperand &init : linalgOp.getDpsInitsMutable()) {
      if (auto allocOp = init.get().getDefiningOp<memref::AllocOp>()) {
        allocOp->setAttr("d2m.blocking_map", blockingMap);
      }
    }
    linalgOp->removeAttr("d2m.blocking_map");
  }
}

class D2MInsertScratchBuffers
    : public impl::D2MInsertScratchBuffersBase<D2MInsertScratchBuffers> {
  using D2MInsertScratchBuffersBase::D2MInsertScratchBuffersBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Collect generics.
    SmallVector<GenericOp> genericsToProcess;
    moduleOp.walk(
        [&](GenericOp genericOp) { genericsToProcess.push_back(genericOp); });

    for (GenericOp genericOp : genericsToProcess) {
      addScratchToGeneric(genericOp);
      transferBlockingMaps(genericOp);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
