// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"
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

// Fallback scratch buffer size in bytes, used as an upper bound, and when the
// DST packing analysis does not produce results for a given generic.
constexpr size_t kFallbackScratchSizeBytes = 128 * 1024; // 128KB

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

// Compute the number of scratch tiles needed for a d2m.generic. Sums the
// per-result numTilesPerResult reported by the DST packing analysis for each
// top-level linalg.generic in the region, skipping ops the analysis did not
// size. Returns kFallbackScratchSizeBytes / tileSizeBytes when the analysis
// produces no usable information. The result is capped at that fallback.
static size_t
computeScratchNumTiles(GenericOp genericOp, ttcore::TileType tileType,
                       const utils::DstRegisterAnalysis &dstAnalysis) {
  const size_t tileSizeBytes = tileType.getSizeBytes();
  const size_t fallbackNumTiles =
      std::max<size_t>(kFallbackScratchSizeBytes / tileSizeBytes, 1);

  const utils::DSTPackingInfo *packingInfo = dstAnalysis.lookup(genericOp);
  if (!packingInfo) {
    return fallbackNumTiles;
  }
  const utils::DSTPackingRegionInfo *regionInfo =
      packingInfo->lookup(&genericOp.getRegion(0));
  if (!regionInfo) {
    return fallbackNumTiles;
  }

  // Sum each top-level linalg.generic's own numTilesPerResult. Per-op entries
  // are looked up by the linalg's single output value; ops not covered by the
  // analysis are skipped and contribute 0 tiles.
  size_t analysisNumTiles = 0;
  for (linalg::GenericOp linalgOp :
       genericOp.getRegion(0).front().getOps<linalg::GenericOp>()) {
    auto it = regionInfo->perResult.find(linalgOp.getOutputs().front());
    if (it == regionInfo->perResult.end()) {
      continue;
    }
    analysisNumTiles += static_cast<size_t>(it->second.numTilesPerResult);
  }
  if (analysisNumTiles == 0) {
    return fallbackNumTiles;
  }
  return std::min(analysisNumTiles, fallbackNumTiles);
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
static void addScratchToGeneric(GenericOp genericOp,
                                const utils::DstRegisterAnalysis &dstAnalysis) {
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

  // Find a tiled input to derive the tile type.
  MemRefType refMemRefType = findTiledInputType(genericOp);
  if (!refMemRefType) {
    return;
  }

  ttcore::TileType tileType = getTileType(refMemRefType);

  // Calculate number of scratch tiles using the DST packing analysis.
  size_t numTiles = computeScratchNumTiles(genericOp, tileType, dstAnalysis);

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

class D2MInsertScratchBuffers
    : public impl::D2MInsertScratchBuffersBase<D2MInsertScratchBuffers> {
  using D2MInsertScratchBuffersBase::D2MInsertScratchBuffersBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Run DST packing analysis on the module to compute per-generic scratch
    // size estimates.
    utils::DstRegisterAnalysis dstAnalysis(moduleOp);

    SmallVector<GenericOp> genericsToProcess;
    moduleOp.walk(
        [&](GenericOp genericOp) { genericsToProcess.push_back(genericOp); });

    for (GenericOp genericOp : genericsToProcess) {
      transferBlockingMaps(genericOp);
      addScratchToGeneric(genericOp, dstAnalysis);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
