// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERSCRATCHALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Calculate total number of elements in a statically-shaped memref.
static int64_t getNumElements(MemRefType memrefType) {
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    assert(dim != ShapedType::kDynamic && "scratch memrefs must be static");
    numElements *= dim;
  }
  return numElements;
}

// Information about a single scratch allocation.
struct ScratchAllocationInfo {
  ScratchAllocateOp op;
  int64_t slotId;
  int64_t numElements;   // Number of elements (tiles) requested.
  int64_t elementOffset; // Starting element offset in scratch buffer.
};

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
      allocations.push_back({allocOp, static_cast<int64_t>(allocOp.getSlot()),
                             getNumElements(memrefType),
                             /*elementOffset=*/0});
    });

    if (allocations.empty()) {
      scratchInit.erase();
      return success();
    }

    // Sort by slot ID for deterministic layout.
    llvm::sort(allocations, [](const auto &a, const auto &b) {
      return a.slotId < b.slotId;
    });

    // Compute sequential element offsets.
    int64_t currentOffset = 0;
    for (auto &info : allocations) {
      assert(info.numElements > 0 && "scratch allocation must be non-empty");
      info.elementOffset = currentOffset;
      currentOffset += info.numElements;
    }

    // Verify allocations fit in the scratch buffer.
    int64_t scratchCapacity = getNumElements(scratchMemRefType);
    if (currentOffset > scratchCapacity) {
      return genericOp.emitOpError()
             << "total scratch allocations (" << currentOffset
             << " elements) exceed scratch buffer capacity (" << scratchCapacity
             << " elements)";
    }

    // The scratch buffer may have been reshaped by D2MAllocate (e.g. from
    // [1, N] to a multi-row shape when aliased with an existing shard CB).
    // Collapse it to a 1-D view so that we can always carve out sub-allocations
    // as simple 1-D subviews regardless of the source shape.
    Value flatScratch = flattenScratch(scratchMemRef, scratchInit);

    // Replace each scratch_allocate with a 1-D subview of the flattened
    // scratch buffer.
    for (auto &info : allocations) {
      replaceScratchAllocate(info, flatScratch);
    }

    // Erase the scratch_init anchor op now that all allocates are lowered.
    scratchInit.erase();

    return success();
  }

  /// Flatten the scratch memref to a 1-D memref via memref.collapse_shape if
  /// needed. Returns the original value if it is already 1-D. The collapse
  /// op is inserted right before the scratch_init anchor so that it dominates
  /// all scratch_allocate users.
  Value flattenScratch(Value scratchMemRef, ScratchInitOp scratchInit) {
    auto sourceType = mlir::cast<MemRefType>(scratchMemRef.getType());
    if (sourceType.getRank() <= 1) {
      return scratchMemRef;
    }

    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices allDims =
        llvm::to_vector(llvm::seq<int64_t>(0, sourceType.getRank()));
    reassociation.push_back(std::move(allDims));

    assert(memref::CollapseShapeOp::isGuaranteedCollapsible(sourceType,
                                                            reassociation) &&
           "scratch buffer must be contiguous to be collapsible");

    OpBuilder builder(scratchInit);
    return builder
        .create<memref::CollapseShapeOp>(scratchInit.getLoc(), scratchMemRef,
                                         reassociation)
        .getResult();
  }

  /// Replace a scratch_allocate with a 1-D subview of the (already flattened)
  /// scratch memref, followed by an expand_shape if the requested type is
  /// multi-dimensional.
  ///
  /// Each scratch_allocate requests a memref with numElements total tiles.
  /// We emit:
  ///   1. subview [offset][M][1] : memref<Kx...> -> memref<M>  (1D slice)
  ///   2. expand_shape memref<M> [[0,1,...,rank-1]] -> memref<requested shape>
  ///      (only if the requested type has rank > 1)
  void replaceScratchAllocate(ScratchAllocationInfo &info,
                              Value scratchMemRef) {
    ScratchAllocateOp allocOp = info.op;
    auto requestedType = mlir::cast<MemRefType>(allocOp.getResult().getType());

    OpBuilder builder(allocOp);
    Location loc = allocOp.getLoc();

    auto sourceType = mlir::cast<MemRefType>(scratchMemRef.getType());
    assert(sourceType.getRank() == 1 &&
           "scratch memref must be flattened to 1-D before subview");

    // Extract a 1-D slice of the flattened scratch buffer.
    SmallVector<int64_t> flatShape = {info.numElements};
    SmallVector<int64_t> staticOffsets = {info.elementOffset};
    SmallVector<int64_t> staticSizes = {info.numElements};
    SmallVector<int64_t> staticStrides = {1};

    // Infer the correct result type. For non-zero offsets, the inferred type
    // includes a strided layout (e.g. strided<[1], offset: N>).
    auto inferredType = memref::SubViewOp::inferResultType(
        sourceType, staticOffsets, staticSizes, staticStrides);

    SmallVector<OpFoldResult> offsets = {
        builder.getIndexAttr(info.elementOffset)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(info.numElements)};
    SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1)};

    auto subviewOp = builder.create<memref::SubViewOp>(
        loc, mlir::cast<MemRefType>(inferredType), scratchMemRef, offsets,
        sizes, strides);

    Value result = subviewOp.getResult();

    // If the requested type is multi-dimensional, reshape the flat 1D slice.
    // We must compute the correct expanded type (preserving any strided layout
    // from the subview) rather than using requestedType directly.
    if (requestedType.getRank() > 1) {
      SmallVector<ReassociationIndices> reassociation;
      ReassociationIndices allDims;
      for (int64_t i = 0; i < requestedType.getRank(); ++i) {
        allDims.push_back(i);
      }
      reassociation.push_back(allDims);

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
