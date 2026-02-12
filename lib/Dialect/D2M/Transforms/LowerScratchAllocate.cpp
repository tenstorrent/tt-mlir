// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/CBUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "d2m-lower-scratch-allocate"

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

// Information about a single scratch allocation, including its liveness range.
struct ScratchAllocationInfo {
  ScratchAllocateOp op;
  int64_t slotId;
  int64_t numElements;   // Number of elements (tiles) requested.
  int64_t elementOffset; // Starting element offset in scratch buffer.

  // Liveness range: positions of the definition and last use within the
  // top-level operation ordering of the entry block.
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
  // rank-reducing subviews of the scratch memref obtained from
  // get_scratch_from_cb.
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

    // Create a CB for the scratch operand using the CB rework utilities.
    // This gives the scratch buffer a proper CB port number so that
    // downstream compute APIs (unpack/pack) can address it.
    // The CB must be created even if there are no scratch_allocate ops,
    // because the scratch memref.alloc in the region needs a CB association
    // for D2MToTTKernel conversion.
    int64_t scratchInputIdx = scratchInputsAttr[0];
    Block &block = region.front();

    IRRewriter rewriter(genericOp->getContext());
    CBCache cbCache;
    PortCounter portCounters;
    Value scratchCB = getOrCreateCB(genericOp, region, scratchInputIdx,
                                    rewriter, cbCache, portCounters);

    // Collect all scratch_allocate ops in this region.
    SmallVector<ScratchAllocationInfo> allocations;
    region.walk([&](ScratchAllocateOp allocOp) {
      auto memrefType = mlir::cast<MemRefType>(allocOp.getResult().getType());
      allocations.push_back({allocOp, static_cast<int64_t>(allocOp.getSlot()),
                             getNumElements(memrefType), /*elementOffset=*/0});
    });

    if (allocations.empty()) {
      return success();
    }

    Block &block = region.front();

    // Sort descending by size so larger allocations are placed first, giving
    // the best chance for smaller ones to be packed inside them.
    llvm::sort(allocations, [](const auto &a, const auto &b) {
      return a.numElements > b.numElements;
    });

    computeLiveness(allocations, block, region);
    int64_t peakUsage = assignOffsets(allocations);

    // Get the scratch operand value (CB block arg or memref.alloc).
    int64_t scratchInputIdx = scratchInputsAttr[0];
    Value scratchValue =
        d2m::GenericOp::getOperandAlloc(region, scratchInputIdx);

    OpBuilder builder(&block, block.begin());
    builder.setInsertionPointAfterValue(scratchCB);
    auto scratchFromCBOp =
        builder.create<GetScratchFromCBOp>(genericOp.getLoc(), scratchCB);
    Value scratchMemRef = scratchFromCBOp.getResult();
    auto scratchMemRefType = mlir::cast<MemRefType>(scratchMemRef.getType());

    // Verify allocations fit in the scratch buffer.
    int64_t scratchCapacity = getNumElements(scratchMemRefType);
    if (peakUsage > scratchCapacity) {
      return genericOp.emitOpError()
             << "total scratch allocations (" << peakUsage
             << " elements) exceed scratch buffer capacity (" << scratchCapacity
             << " elements)";
    }

    // Replace each scratch_allocate with a rank-reducing subview.
    for (auto &info : allocations) {
      replaceScratchAllocate(info, scratchMemRef);
    }

    return success();
  }

  // Compute liveness ranges for scratch allocations.
  //
  // Each allocation's live range spans from its definition to its last use.
  void computeLiveness(SmallVectorImpl<ScratchAllocationInfo> &allocations,
                       Block &block, Region &region) {
    // Assign sequential position indices to top-level operations in the block.
    DenseMap<Operation *, int64_t> opPositions;
    int64_t pos = 0;
    for (Operation &op : block) {
      opPositions[&op] = pos++;
    }

    // Helper: walk up to find the enclosing operation that lives directly in
    // the region's entry block.
    auto getTopLevelOp = [&](Operation *op) -> Operation * {
      while (op->getParentRegion() != &region) {
        op = op->getParentOp();
      }
      return op;
    };

    for (auto &info : allocations) {
      Operation *topLevelDef = getTopLevelOp(info.op.getOperation());
      info.startPosition = opPositions[topLevelDef];
      info.endPosition = info.startPosition;

      for (OpOperand &use : info.op.getResult().getUses()) {
        Operation *topLevelUser = getTopLevelOp(use.getOwner());
        int64_t userPos = opPositions[topLevelUser];
        info.endPosition = std::max(info.endPosition, userPos);
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

  /// Replace a scratch_allocate with a rank-reducing subview of the scratch CB,
  /// followed by an expand_shape if the requested type is multi-dimensional.
  ///
  /// The scratch buffer has shape [1, N] from AddScratchInputs.
  /// Each scratch_allocate requests a memref with numElements total tiles.
  /// We emit:
  ///   1. subview [0, offset][1, M][1, 1] : memref<1xN> -> memref<M>  (flat 1D)
  ///   2. expand_shape memref<M> [[0,1,...,rank-1]] -> memref<requested shape>
  ///      (only if the requested type has rank > 1)
  void replaceScratchAllocate(ScratchAllocationInfo &info,
                              Value scratchMemRef) {
    ScratchAllocateOp allocOp = info.op;
    auto requestedType = mlir::cast<MemRefType>(allocOp.getResult().getType());

    OpBuilder builder(allocOp);
    Location loc = allocOp.getLoc();

    // Always extract a flat 1D slice from the 2D scratch CB.
    SmallVector<int64_t> flatShape = {info.numElements};
    SmallVector<int64_t> staticOffsets = {0, info.elementOffset};
    SmallVector<int64_t> staticSizes = {1, info.numElements};
    SmallVector<int64_t> staticStrides = {1, 1};

    // Infer the correct rank-reduced result type. For non-zero offsets, the
    // inferred type includes a strided layout (e.g. strided<[1], offset: N>).
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
