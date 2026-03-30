// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

// Check if the generic already has a scratch alloc in its region.
static bool hasScratchAlloc(GenericOp genericOp) {
  bool found = false;
  for (Region &region : genericOp->getRegions()) {
    region.walk([&](memref::AllocOp allocOp) {
      if (allocOp->hasAttr("d2m.scratch")) {
        found = true;
      }
    });
  }
  return found;
}

// Add a local scratch allocation inside a d2m.generic op's region.
// The scratch buffer is a memref.alloc with CBLayoutAttr and a "d2m.scratch"
// marker attribute. It is placed after existing operand allocs in the region
// and will be picked up by the Allocate pass for L1 address planning, then
// hoisted by HoistCBAllocs as an additionalArg.
static LogicalResult addScratchToGeneric(GenericOp genericOp) {
  // Skip if this generic already has a scratch alloc.
  if (hasScratchAlloc(genericOp)) {
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

  // Find a tiled input to get the tile type.
  MemRefType refMemRefType = findTiledInputType(genericOp);
  if (!refMemRefType) {
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

  // Shard shape for the local CB: [1, numTiles].
  SmallVector<int64_t> scratchShardShape = {1, static_cast<int64_t>(numTiles)};

  // Create CBLayoutAttr for the scratch buffer (single-buffered).
  auto cbLayout =
      ttcore::CBLayoutAttr::get(scratchShardShape, tileType, /*buffers=*/1);

  auto l1MemorySpace = ttcore::MemorySpaceAttr::get(
      genericOp.getContext(), ttcore::MemorySpace::DeviceL1);

  MemRefType scratchMemRefType =
      MemRefType::get(scratchShardShape, tileType, cbLayout, l1MemorySpace);

  // Insert the scratch alloc inside the generic region, after the last
  // operand alloc (tensor.empty or memref.alloc).
  if (genericOp.getNumRegions() == 0) {
    return failure();
  }
  Region &region = genericOp.getRegion(0);
  if (region.empty()) {
    return failure();
  }
  Block &block = region.front();

  // Find insertion point: after the last tensor.empty/memref.alloc that
  // corresponds to an operand (i.e., after all input+output allocs).
  unsigned numOperands =
      genericOp.getInputs().size() + genericOp.getOutputs().size();
  unsigned allocCount = 0;
  Operation *insertAfter = nullptr;
  for (Operation &op : block) {
    if (mlir::isa<mlir::tensor::EmptyOp, memref::AllocOp>(&op) &&
        !op.hasAttr("d2m.scratch")) {
      ++allocCount;
      insertAfter = &op;
      if (allocCount == numOperands) {
        break;
      }
    }
  }

  OpBuilder builder(&block, insertAfter ? std::next(insertAfter->getIterator())
                                        : block.begin());
  auto scratchAlloc =
      builder.create<memref::AllocOp>(genericOp.getLoc(), scratchMemRefType);
  scratchAlloc->setAttr("d2m.scratch", builder.getUnitAttr());

  return success();
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
      // addScratchToGeneric returns failure for generics that don't need
      // scratch (not an error). We intentionally ignore the result.
      (void)addScratchToGeneric(genericOp);
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
