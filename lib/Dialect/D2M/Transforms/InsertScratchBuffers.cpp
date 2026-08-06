// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
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

// Fixed scratch buffer size used for every fused d2m.generic that needs
// intermediate spills.
constexpr size_t kScratchSizeBytes = 128 * 1024;

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

// Count the number of linalg.generic ops in the first region of a
// d2m.generic. A count greater than one indicates a fused kernel whose
// intermediate results may need to be spilled to a scratch buffer.
static unsigned countLinalgGenerics(GenericOp genericOp) {
  if (genericOp.getNumRegions() == 0) {
    return 0;
  }

  unsigned linalgCount = 0;
  genericOp.getRegion(0).walk([&](linalg::GenericOp) { ++linalgCount; });
  return linalgCount;
}

// Return kScratchSizeBytes expressed in tiles of `tileType` (at least 1).
static size_t computeScratchNumTiles(ttcore::TileType tileType) {
  return std::max<size_t>(kScratchSizeBytes / tileType.getSizeBytes(), 1);
}

// Transfer d2m.blocking_map attributes from inner linalg ops (set during
// elementwise fusion) to their output memref.alloc ops, so that the allocator
// can read the map directly from the alloc.
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
  unsigned linalgCount = countLinalgGenerics(genericOp);
  if (linalgCount <= 1) {
    return;
  }

  MemRefType refMemRefType = findTiledReferenceType(genericOp);
  if (!refMemRefType) {
    return;
  }

  ttcore::TileType tileType = getTileType(refMemRefType);

  size_t numTiles = computeScratchNumTiles(tileType);

  // Build scratch shard shape: [1, numTiles].
  SmallVector<int64_t> scratchShardShape = {1, static_cast<int64_t>(numTiles)};

  auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
      genericOp.getContext(), ttcore::MemorySpace::DeviceL1);

  // Shard memref type for the scratch buffer inside the region.
  MemRefType scratchShardMemRefType = MemRefType::get(
      scratchShardShape, tileType, MemRefLayoutAttrInterface{}, l1MemorySpace);

  // Insert memref.alloc + scratch_init at the start of the region body.
  Block &block = genericOp.getRegion(0).front();
  OpBuilder builder(&block, block.begin());

  auto scratchAlloc = builder.create<memref::AllocOp>(genericOp.getLoc(),
                                                      scratchShardMemRefType);
  scratchAlloc->setAttr("d2m.scratch_buffer", builder.getUnitAttr());
  builder.create<ScratchInitOp>(genericOp.getLoc(), scratchAlloc.getResult());
}

static void addTopkIndexBuffers(GenericOp genericOp) {
  if (genericOp.getNumRegions() == 0) {
    return;
  }

  // At most one generate_indices topk_block per generic.
  TopkBlockOp topkBlock = nullptr;
  genericOp.getRegion(0).walk([&](TopkBlockOp op) {
    if (op.getGenerateIndices()) {
      topkBlock = op;
    }
  });
  if (!topkBlock) {
    return;
  }

  // Post-bufferization only; nothing to allocate while still on tensors.
  auto inputType =
      mlir::dyn_cast<MemRefType>(topkBlock.getInputValues().getType());
  auto indicesType =
      mlir::dyn_cast<MemRefType>(topkBlock.getOutIndices().getType());
  if (!inputType || !indicesType) {
    return;
  }

  auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
      genericOp.getContext(), ttcore::MemorySpace::DeviceL1);

  Block &block = genericOp.getRegion(0).front();
  OpBuilder builder(&block, block.begin());

  auto allocScratch = [&](ArrayRef<int64_t> bufShape, StringRef roleAttr) {
    auto bufType = MemRefType::get(bufShape, indicesType.getElementType(),
                                   MemRefLayoutAttrInterface{}, l1MemorySpace);
    auto allocOp = builder.create<memref::AllocOp>(topkBlock.getLoc(), bufType);
    allocOp->setAttr("d2m.scratch_buffer", builder.getUnitAttr());
    allocOp->setAttr(roleAttr, builder.getUnitAttr());
    // Without this the canonicalizer erases the still-unused alloc.
    builder.create<ScratchInitOp>(topkBlock.getLoc(), allocOp.getResult());
  };

  // One index tile per value tile, plus the lane pattern they all derive from.
  allocScratch(inputType.getShape(), utils::kTopkIndexBufferAttr);
  allocScratch({utils::kTopkLaneTileRows, 1}, utils::kTopkLaneBufferAttr);
}

class D2MInsertScratchBuffers
    : public impl::D2MInsertScratchBuffersBase<D2MInsertScratchBuffers> {
  using D2MInsertScratchBuffersBase::D2MInsertScratchBuffersBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    SmallVector<GenericOp> genericsToProcess;
    moduleOp.walk(
        [&](GenericOp genericOp) { genericsToProcess.push_back(genericOp); });

    for (GenericOp genericOp : genericsToProcess) {
      transferBlockingMaps(genericOp);
      addScratchToGeneric(genericOp);
      addTopkIndexBuffers(genericOp);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
