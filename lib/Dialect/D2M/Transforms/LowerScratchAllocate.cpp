// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"

#include <limits>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERSCRATCHALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Information about a single scratch allocation, including its liveness range.
struct ScratchAllocationInfo {
  ScratchAllocateOp op;
  int64_t slotId;
  int64_t numElements;   // Number of elements (tiles) requested.
  int64_t elementOffset; // Starting element offset in scratch buffer.

  // Liveness range, as positions in a region-wide pre-order walk over the
  // owning d2m.generic, taken across the slot's uses (first/last store/load).
  int64_t startPosition = -1;
  int64_t endPosition = -1;

  // Whether this allocation was packed inside a larger, non-conflicting
  // allocation's footprint by the offset assignment heuristic.
  bool packed = false;
};

// Returns true if two live ranges overlap.
static bool livesOverlap(const ScratchAllocationInfo &a,
                         const ScratchAllocationInfo &b) {
  return a.startPosition <= b.endPosition && b.startPosition <= a.endPosition;
}

class D2MLowerScratchAllocatePass
    : public impl::D2MLowerScratchAllocateBase<D2MLowerScratchAllocatePass> {
public:
  using D2MLowerScratchAllocateBase::D2MLowerScratchAllocateBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    WalkResult result = moduleOp.walk([&](GenericOp genericOp) {
      if (failed(processGeneric(genericOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

private:
  // Process a single d2m.generic, lowering all scratch_allocate ops to
  // rank-reducing subviews of the scratch memref obtained from scratch_init.
  LogicalResult processGeneric(GenericOp genericOp) {
    if (genericOp.getNumRegions() == 0) {
      return success();
    }
    Region &region = genericOp.getRegion(0);
    if (region.empty()) {
      return success();
    }

    // Find the scratch_init op in the region.
    ScratchInitOp scratchInit = nullptr;
    region.walk([&](ScratchInitOp op) { scratchInit = op; });

    if (!scratchInit) {
      return success();
    }

    Value scratchMemRef = scratchInit.getScratch();
    auto scratchMemRefType = mlir::cast<MemRefType>(scratchMemRef.getType());

    // Collect all scratch_allocate ops in this region.
    SmallVector<ScratchAllocationInfo> allocations;
    region.walk([&](ScratchAllocateOp allocOp) {
      auto memrefType = mlir::cast<MemRefType>(allocOp.getResult().getType());
      assert(!llvm::is_contained(memrefType.getShape(), ShapedType::kDynamic) &&
             "scratch memrefs must be static");
      allocations.push_back(
          {allocOp, static_cast<int64_t>(allocOp.getSlot()),
           ttmlir::utils::volume<int64_t>(memrefType.getShape()),
           /*elementOffset=*/0});
    });

    if (allocations.empty()) {
      scratchInit.erase();
      return success();
    }

    // Sort descending by size so larger allocations are placed first, giving
    // the best chance for smaller ones to be packed inside them.
    llvm::sort(allocations, [](const auto &a, const auto &b) {
      if (a.numElements != b.numElements) {
        return a.numElements > b.numElements;
      }
      return a.slotId < b.slotId;
    });

    computeLiveness(allocations, region);
    int64_t peakUsage = assignOffsets(allocations);

    // Verify allocations fit in the scratch buffer.
    assert(!llvm::is_contained(scratchMemRefType.getShape(),
                               ShapedType::kDynamic) &&
           "scratch memrefs must be static");
    int64_t scratchCapacity =
        ttmlir::utils::volume<int64_t>(scratchMemRefType.getShape());
    if (peakUsage > scratchCapacity) {
      return genericOp.emitOpError()
             << "peak scratch usage (" << peakUsage
             << " elements) exceed scratch buffer capacity (" << scratchCapacity
             << " elements)";
    }

    // Replace each scratch_allocate with a rank-reducing subview.
    for (auto &info : allocations) {
      replaceScratchAllocate(info, scratchMemRef);
    }

    // Erase the scratch_init anchor op now that all allocates are lowered.
    scratchInit.erase();

    return success();
  }

  // Compute liveness ranges for scratch allocations.
  //
  // Positions come from a region-wide pre-order walk so that ops nested in
  // `scratch_space_loop` nests get distinct indices.
  void computeLiveness(SmallVectorImpl<ScratchAllocationInfo> &allocations,
                       Region &region) {
    DenseMap<Operation *, int64_t> opPositions;
    int64_t pos = 0;
    region.walk<WalkOrder::PreOrder>(
        [&](Operation *op) { opPositions[op] = pos++; });

    for (auto &info : allocations) {
      info.startPosition = std::numeric_limits<int64_t>::max();
      info.endPosition = std::numeric_limits<int64_t>::min();
      for (OpOperand &use : info.op.getResult().getUses()) {
        auto it = opPositions.find(use.getOwner());
        assert(it != opPositions.end() && "use is outside scratch region");
        info.startPosition = std::min(info.startPosition, it->second);
        info.endPosition = std::max(info.endPosition, it->second);
      }
      if (info.startPosition > info.endPosition) {
        // Unused slot: pin the range to the def so it still gets an offset.
        Operation *defOp = info.op.getOperation();
        assert(opPositions.contains(defOp) && "def is outside scratch region");
        info.startPosition = info.endPosition = opPositions[defOp];
      }
    }
  }

  // Assign element offsets using a first-fit-decreasing heuristic.
  //
  // Allocations are already sorted descending by size. Each "outer" allocation
  // reserves a region of scratch memory. Smaller allocations whose live ranges
  // do not overlap with the outer allocation are packed inside its footprint,
  // reusing the same memory at different sub-offsets.
  //
  // Returns the peak scratch usage (total elements required).
  int64_t assignOffsets(SmallVectorImpl<ScratchAllocationInfo> &allocations) {
    int64_t currentOffset = 0;
    for (size_t i = 0; i < allocations.size(); ++i) {
      auto &outer = allocations[i];
      assert(outer.numElements > 0 && "scratch allocation must be non-empty");
      if (outer.packed) {
        continue;
      }

      outer.elementOffset = currentOffset;
      int64_t innerOffset = currentOffset;

      // Try to pack smaller, non-conflicting allocations inside this one.
      for (size_t j = i + 1; j < allocations.size(); ++j) {
        auto &inner = allocations[j];
        if (inner.packed) {
          continue;
        }

        bool fits = (innerOffset + inner.numElements) <=
                    (currentOffset + outer.numElements);
        if (!livesOverlap(outer, inner) && fits) {
          inner.packed = true;
          inner.elementOffset = innerOffset;
          innerOffset += inner.numElements;
        }
      }

      currentOffset += outer.numElements;
    }
    return currentOffset;
  }

  // Replace a scratch_allocate with a rank-reducing subview of the scratch
  // memref, followed by an expand_shape if the requested type is
  // multi-dimensional.
  //
  // The scratch buffer has shape [1, N] from InsertScratchBuffers.
  // Each scratch_allocate requests a memref with numElements total tiles.
  // We emit:
  //   1. subview [0, offset][1, M][1, 1] : memref<1xN> -> memref<M>  (flat 1D)
  //   2. expand_shape memref<M> [[0,1,...,rank-1]] -> memref<requested shape>
  //      (only if the requested type has rank > 1)
  void replaceScratchAllocate(ScratchAllocationInfo &info,
                              Value scratchMemRef) {
    ScratchAllocateOp allocOp = info.op;
    auto requestedType = mlir::cast<MemRefType>(allocOp.getResult().getType());

    OpBuilder builder(allocOp);
    Location loc = allocOp.getLoc();

    SmallVector<int64_t> flatShape = {info.numElements};
    SmallVector<int64_t> staticOffsets = {0, info.elementOffset};
    SmallVector<int64_t> staticSizes = {1, info.numElements};
    SmallVector<int64_t> staticStrides = {1, 1};

    // For non-zero offsets the inferred type carries a strided layout
    // (e.g. strided<[1], offset: N>), which the requested type does not.
    auto sourceType = mlir::cast<MemRefType>(scratchMemRef.getType());
    auto inferredType = memref::SubViewOp::inferRankReducedResultType(
        flatShape, sourceType, staticOffsets, staticSizes, staticStrides);

    SmallVector<OpFoldResult> offsets = {
        builder.getIndexAttr(0), builder.getIndexAttr(info.elementOffset)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(info.numElements)};
    SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                         builder.getIndexAttr(1)};

    auto subviewOp = builder.create<memref::SubViewOp>(
        loc, mlir::cast<MemRefType>(inferredType), scratchMemRef, offsets,
        sizes, strides);

    Value result = subviewOp.getResult();

    if (requestedType.getRank() > 1) {
      ReassociationIndices allDims;
      for (int64_t i = 0; i < requestedType.getRank(); ++i) {
        allDims.push_back(i);
      }
      SmallVector<ReassociationIndices> reassociation = {allDims};

      auto subviewType = mlir::cast<MemRefType>(result.getType());
      auto expandedType = memref::ExpandShapeOp::computeExpandedType(
          subviewType, requestedType.getShape(), reassociation);
      assert(succeeded(expandedType) && "failed to compute expanded type");

      result = builder.create<memref::ExpandShapeOp>(loc, *expandedType, result,
                                                     reassociation);
    }

    allocOp.getResult().replaceAllUsesWith(result);
    allocOp.erase();
  }
};

} // namespace
} // namespace mlir::tt::d2m
