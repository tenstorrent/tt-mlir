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

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERSCRATCHALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Calculate total number of elements in a statically-shaped memref.
static int64_t getNumElements(MemRefType memrefType) {
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    assert(dim != ShapedType::kDynamic && "scratch memrefs must be static");
    numElements *= dim;
  }
  return numElements;
}

/// Information about a single scratch allocation.
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
  /// Process a single d2m.generic, lowering all scratch_allocate ops to
  /// rank-reducing subviews of the scratch CB.
  LogicalResult processGeneric(GenericOp genericOp) {
    auto scratchInputsAttr = genericOp.getScratchInputsAttr();
    if (!scratchInputsAttr || scratchInputsAttr.empty()) {
      return success();
    }

    if (genericOp.getNumRegions() == 0) {
      return success();
    }
    Region &region = genericOp.getRegion(0);
    if (region.empty()) {
      return success();
    }
    Block &block = region.front();

    // Collect all scratch_allocate ops in this region.
    SmallVector<ScratchAllocationInfo> allocations;
    region.walk([&](ScratchAllocateOp allocOp) {
      auto memrefType = mlir::cast<MemRefType>(allocOp.getResult().getType());
      allocations.push_back({allocOp, static_cast<int64_t>(allocOp.getSlot()),
                             getNumElements(memrefType),
                             /*elementOffset=*/0});
    });

    if (allocations.empty()) {
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

    // Look up scratch CB block argument.
    int64_t scratchIdx = scratchInputsAttr[0];
    assert(scratchIdx < static_cast<int64_t>(block.getNumArguments()) &&
           "scratch operand index out of bounds");
    Value scratchCB = block.getArgument(scratchIdx);

    auto cbType = mlir::cast<CBType>(scratchCB.getType());
    auto scratchMemRefType = mlir::cast<MemRefType>(cbType.getUnderlying());

    // Verify allocations fit in the scratch buffer.
    int64_t scratchCapacity = getNumElements(scratchMemRefType);
    if (currentOffset > scratchCapacity) {
      return genericOp.emitOpError()
             << "total scratch allocations (" << currentOffset
             << " elements) exceed scratch buffer capacity (" << scratchCapacity
             << " elements)";
    }

    // Determine number of grid dimensions from global vs shard rank.
    Value scratchGlobalOperand = genericOp->getOperand(scratchIdx);
    auto globalMemRefType =
        mlir::cast<MemRefType>(scratchGlobalOperand.getType());
    int64_t numGridDims =
        globalMemRefType.getRank() - scratchMemRefType.getRank();
    assert(numGridDims >= 0 &&
           "global shape must have at least as many dims as shard shape");

    // Create memref.alloc + remote_load at the top of the compute region.
    // D2MConvertLocalLoadStoreOpsToAliasedCBs will later convert this pattern
    // to CB-based access (reserve/push/wait/pop).
    OpBuilder builder(&block, block.begin());
    Location loc = genericOp.getLoc();

    auto scratchAlloc = builder.create<memref::AllocOp>(
        loc, scratchMemRefType, builder.getI64IntegerAttr(64));

    SmallVector<Value> blockIndices;
    for (int64_t i = 0; i < numGridDims; ++i) {
      blockIndices.push_back(builder.create<BlockIndexOp>(loc, i));
    }

    auto remoteLoadOp = builder.create<RemoteLoadOp>(
        loc, scratchMemRefType, scratchAlloc.getResult(), scratchGlobalOperand,
        blockIndices);
    Value scratchMemRef = remoteLoadOp.getResult();

    // Replace each scratch_allocate with a rank-reducing subview.
    for (auto &info : allocations) {
      replaceScratchAllocate(info, scratchMemRef);
    }

    return success();
  }

  /// Replace a scratch_allocate with a rank-reducing subview of the scratch CB.
  ///
  /// The scratch buffer has shape [1, N] from AddScratchInputs.
  /// Each scratch_allocate requests a 1D memref<M x tile>.
  /// We emit: subview [0, offset][1, M][1, 1] : memref<1xN> -> memref<M>
  void replaceScratchAllocate(ScratchAllocationInfo &info,
                              Value scratchMemRef) {
    ScratchAllocateOp allocOp = info.op;
    auto requestedType = mlir::cast<MemRefType>(allocOp.getResult().getType());

    OpBuilder builder(allocOp);
    Location loc = allocOp.getLoc();

    SmallVector<int64_t> staticOffsets = {0, info.elementOffset};
    SmallVector<int64_t> staticSizes = {1, info.numElements};
    SmallVector<int64_t> staticStrides = {1, 1};

    // Infer the correct rank-reduced result type. For non-zero offsets, the
    // inferred type includes a strided layout (e.g. strided<[1], offset: N>).
    auto sourceType = mlir::cast<MemRefType>(scratchMemRef.getType());
    auto inferredType = memref::SubViewOp::inferRankReducedResultType(
        requestedType.getShape(), sourceType, staticOffsets, staticSizes,
        staticStrides);

    SmallVector<OpFoldResult> offsets = {
        builder.getIndexAttr(0), builder.getIndexAttr(info.elementOffset)};
    SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(1),
                                       builder.getIndexAttr(info.numElements)};
    SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                         builder.getIndexAttr(1)};

    auto subviewOp = builder.create<memref::SubViewOp>(
        loc, mlir::cast<MemRefType>(inferredType), scratchMemRef, offsets,
        sizes, strides);

    allocOp.getResult().replaceAllUsesWith(subviewOp.getResult());
    allocOp.erase();
  }
};

} // namespace
} // namespace mlir::tt::d2m
