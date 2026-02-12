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

// Information about a single scratch allocation, including its liveness range
// and set of conflicting allocations.
struct ScratchAllocationInfo {
  ScratchAllocateOp op;
  int64_t slotId;
  int64_t numElements;   // Number of elements (tiles) requested.
  int64_t elementOffset; // Starting element offset in scratch buffer.

  // Liveness range: positions of the definition and last use within the
  // top-level operation ordering of the entry block. A value of -1 indicates
  // that liveness has not been computed yet.
  int64_t startPosition = -1;
  int64_t endPosition = -1;

  // Indices (into the allocations vector) of allocations whose live ranges
  // overlap with this one.
  SmallVector<size_t> conflicts;
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

    Block &block = region.front();

    // Sort by slot ID for deterministic layout.
    llvm::sort(allocations, [](const auto &a, const auto &b) {
      return a.slotId < b.slotId;
    });

    // Compute liveness ranges and conflicts before deciding offsets.
    computeLiveness(allocations, block, region);

    // Compute sequential element offsets.
    // TODO(ckaravasilisTT): Use liveness-based allocation instead of
    // sequential packing to reduce peak scratch usage.
    int64_t currentOffset = 0;
    for (auto &info : allocations) {
      assert(info.numElements > 0 && "scratch allocation must be non-empty");
      info.elementOffset = currentOffset;
      currentOffset += info.numElements;
    }

    // Get the scratch CB block argument and create get_scratch_from_cb.
    int64_t scratchInputIdx = scratchInputsAttr[0];
    Value scratchCBArg = block.getArgument(scratchInputIdx);

    OpBuilder builder(&block, block.begin());
    auto scratchFromCBOp =
        builder.create<GetScratchFromCBOp>(genericOp.getLoc(), scratchCBArg);

    Value scratchMemRef = scratchFromCBOp.getResult();
    auto scratchMemRefType = mlir::cast<MemRefType>(scratchMemRef.getType());

    // Verify allocations fit in the scratch buffer.
    int64_t scratchCapacity = getNumElements(scratchMemRefType);
    if (currentOffset > scratchCapacity) {
      return genericOp.emitOpError()
             << "total scratch allocations (" << currentOffset
             << " elements) exceed scratch buffer capacity (" << scratchCapacity
             << " elements)";
    }

    // Replace each scratch_allocate with a rank-reducing subview.
    for (auto &info : allocations) {
      replaceScratchAllocate(info, scratchMemRef);
    }

    return success();
  }

  // Compute liveness ranges and conflict sets for scratch allocations.
  //
  // Each allocation's live range spans from its definition to its last use,
  // where nested uses (e.g. inside affine.for) are mapped to the enclosing
  // top-level operation in the block. Two allocations conflict if their live
  // ranges overlap.
  void computeLiveness(SmallVectorImpl<ScratchAllocationInfo> &allocations,
                       Block &block, Region &region) {
    // Assign sequential position indices to top-level operations in the block.
    DenseMap<Operation *, int64_t> opPositions;
    int64_t pos = 0;
    for (Operation &op : block) {
      opPositions[&op] = pos++;
    }

    // Helper: walk up to find the enclosing operation that lives directly in
    // the region's entry block. This maps nested uses (e.g. inside affine.for)
    // to the top-level op that contains them.
    auto getTopLevelOp = [&](Operation *op) -> Operation * {
      while (op->getParentRegion() != &region) {
        op = op->getParentOp();
      }
      return op;
    };

    // Compute start and end positions for each allocation.
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

    // Compute conflicts: two allocations conflict if their live ranges overlap.
    for (size_t i = 0; i < allocations.size(); ++i) {
      for (size_t j = i + 1; j < allocations.size(); ++j) {
        bool overlaps =
            allocations[i].startPosition <= allocations[j].endPosition &&
            allocations[j].startPosition <= allocations[i].endPosition;
        if (overlaps) {
          allocations[i].conflicts.push_back(j);
          allocations[j].conflicts.push_back(i);
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "=== Scratch liveness analysis ===\n";
      for (size_t i = 0; i < allocations.size(); ++i) {
        auto &info = allocations[i];
        llvm::dbgs() << "  slot " << info.slotId << ": [" << info.startPosition
                     << ", " << info.endPosition << "] (" << info.numElements
                     << " elements)";
        if (!info.conflicts.empty()) {
          llvm::dbgs() << " conflicts with slots {";
          llvm::interleaveComma(info.conflicts, llvm::dbgs(), [&](size_t idx) {
            llvm::dbgs() << allocations[idx].slotId;
          });
          llvm::dbgs() << "}";
        }
        llvm::dbgs() << "\n";
      }
    });
  }

  // Replace a scratch_allocate with a rank-reducing subview of the scratch
  // memref. The scratch buffer has shape [1, N] from AddScratchInputs.
  // Each scratch_allocate requests a 1D memref<M x tile>.
  // We emit: subview [0, offset][1, M][1, 1] : memref<1xN> -> memref<M>
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
