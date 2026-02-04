// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERSCRATCHALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
} // namespace mlir::tt::d2m

using namespace mlir;
using namespace mlir::tt;

namespace {

/// Calculate total number of elements in a memref.
static int64_t getNumElements(MemRefType memrefType) {
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      return -1;
    }
    numElements *= dim;
  }
  return numElements;
}

/// Information about a single scratch allocation.
struct ScratchAllocationInfo {
  d2m::ScratchAllocateOp op;
  int64_t slotId;
  int64_t numElements;   // Number of elements (tiles) requested
  int64_t elementOffset; // Starting element offset in scratch buffer
};

class D2MLowerScratchAllocatePass
    : public d2m::impl::D2MLowerScratchAllocateBase<
          D2MLowerScratchAllocatePass> {
public:
  using d2m::impl::D2MLowerScratchAllocateBase<
      D2MLowerScratchAllocatePass>::D2MLowerScratchAllocateBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    WalkResult result = moduleOp.walk([&](d2m::GenericOp genericOp) {
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
  LogicalResult processGeneric(d2m::GenericOp genericOp) {
    // Check if this generic has scratch inputs
    auto scratchInputsAttr = genericOp.getScratchInputsAttr();
    if (!scratchInputsAttr || scratchInputsAttr.empty()) {
      return success();
    }

    // Get the compute region (first region)
    if (genericOp.getNumRegions() == 0) {
      return success();
    }
    Region &region = genericOp.getRegion(0);
    if (region.empty()) {
      return success();
    }
    Block &block = region.front();

    // Collect all scratch_allocate ops in this region
    SmallVector<ScratchAllocationInfo> allocations;
    region.walk([&](d2m::ScratchAllocateOp allocOp) {
      ScratchAllocationInfo info;
      info.op = allocOp;
      info.slotId = allocOp.getSlot();
      auto memrefType =
          mlir::dyn_cast<MemRefType>(allocOp.getResult().getType());
      info.numElements = memrefType ? getNumElements(memrefType) : 0;
      info.elementOffset = 0;
      allocations.push_back(info);
    });

    if (allocations.empty()) {
      return success();
    }

    // Sort by slot ID for deterministic ordering
    llvm::sort(allocations, [](const ScratchAllocationInfo &a,
                               const ScratchAllocationInfo &b) {
      return a.slotId < b.slotId;
    });

    // Compute sequential element offsets
    int64_t currentOffset = 0;
    for (ScratchAllocationInfo &info : allocations) {
      if (info.numElements <= 0) {
        info.op.emitOpError() << "invalid element count for scratch allocation";
        return failure();
      }
      info.elementOffset = currentOffset;
      currentOffset += info.numElements;
    }

    // Find the scratch CB block argument
    int64_t scratchOperandIndex = scratchInputsAttr[0];
    if (scratchOperandIndex >= static_cast<int64_t>(block.getNumArguments())) {
      genericOp.emitOpError() << "scratch operand index out of bounds";
      return failure();
    }
    Value scratchCB = block.getArgument(scratchOperandIndex);

    auto cbType = mlir::dyn_cast<d2m::CBType>(scratchCB.getType());
    if (!cbType) {
      genericOp.emitOpError() << "scratch operand is not a CB type";
      return failure();
    }

    auto scratchMemRefType = mlir::dyn_cast<MemRefType>(cbType.getUnderlying());
    if (!scratchMemRefType) {
      genericOp.emitOpError() << "scratch CB does not contain a memref type";
      return failure();
    }

    // Verify total fits in scratch buffer
    int64_t scratchCapacity = getNumElements(scratchMemRefType);
    if (scratchCapacity > 0 && currentOffset > scratchCapacity) {
      genericOp.emitOpError() << "total scratch allocations (" << currentOffset
                              << " elements) exceed scratch buffer capacity ("
                              << scratchCapacity << " elements)";
      return failure();
    }

    // Get the global scratch operand (the memref passed to the generic)
    Value scratchGlobalOperand = genericOp->getOperand(scratchOperandIndex);

    // Get the global shape to determine the number of grid dimensions
    auto globalMemRefType =
        mlir::dyn_cast<MemRefType>(scratchGlobalOperand.getType());
    if (!globalMemRefType) {
      genericOp.emitOpError() << "scratch operand is not a memref type";
      return failure();
    }

    // Global shape is [GridY, GridX, ShardY, ShardX] (4D for 2D grid)
    // We need grid indices for remote_load
    int64_t globalRank = globalMemRefType.getRank();
    int64_t shardRank = scratchMemRefType.getRank();
    int64_t numGridDims = globalRank - shardRank;

    if (numGridDims < 0) {
      genericOp.emitOpError()
          << "scratch global shape has fewer dims than shard shape";
      return failure();
    }

    // Create a local memref.alloc for scratch access, then use remote_load to
    // associate it with the scratch operand. The
    // D2MConvertLocalLoadStoreOpsToAliasedCBs pass will later convert this to
    // CB-based access (reserve/push/wait/pop).
    OpBuilder builder(&block, block.begin());
    auto scratchAlloc = builder.create<memref::AllocOp>(
        genericOp.getLoc(), scratchMemRefType,
        builder.getI64IntegerAttr(64) /*alignment*/);

    // Create block indices for remote_load
    SmallVector<Value> blockIndices;
    for (int64_t i = 0; i < numGridDims; ++i) {
      blockIndices.push_back(
          builder.create<d2m::BlockIndexOp>(genericOp.getLoc(), i));
    }

    // Create remote_load to "load" scratch into local buffer.
    // For local operands, this is essentially an alias, but it establishes
    // the association needed for D2MConvertLocalLoadStoreOpsToAliasedCBs.
    auto remoteLoadOp = builder.create<d2m::RemoteLoadOp>(
        genericOp.getLoc(), scratchMemRefType, scratchAlloc.getResult(),
        scratchGlobalOperand, blockIndices);
    Value scratchMemRef = remoteLoadOp.getResult();

    // Replace each scratch_allocate with a subview
    for (ScratchAllocationInfo &info : allocations) {
      if (failed(
              replaceScratchAllocate(info, scratchMemRef, scratchMemRefType))) {
        return failure();
      }
    }

    return success();
  }

  LogicalResult replaceScratchAllocate(ScratchAllocationInfo &info,
                                       Value scratchMemRef,
                                       MemRefType scratchMemRefType) {
    d2m::ScratchAllocateOp allocOp = info.op;
    auto resultType = mlir::dyn_cast<MemRefType>(allocOp.getResult().getType());
    if (!resultType) {
      return allocOp.emitOpError() << "result is not a memref type";
    }

    OpBuilder builder(allocOp);
    Location loc = allocOp.getLoc();

    // Scratch buffer has shape [1, N] tiles from AddScratchInputs.
    // We use memref.subview with rank-reduction to extract the requested slice.
    //
    // For a 1D result [M]: subview [0, offset][1, M][1, 1] -> [M]
    // For a 2D result [A, B] where A*B fits: same approach but may need
    // explicit result type to handle strided layouts.

    ArrayRef<int64_t> scratchShape = scratchMemRefType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    if (scratchShape.size() == 2) {
      // Scratch is [1, N]. We take a slice [0, offset][1, numElements][1, 1].
      offsets.push_back(builder.getIndexAttr(0));
      offsets.push_back(builder.getIndexAttr(info.elementOffset));
      sizes.push_back(builder.getIndexAttr(1));
      sizes.push_back(builder.getIndexAttr(info.numElements));
      strides.push_back(builder.getIndexAttr(1));
      strides.push_back(builder.getIndexAttr(1));
    } else if (scratchShape.size() == 1) {
      // Scratch is [N]. Direct slice.
      offsets.push_back(builder.getIndexAttr(info.elementOffset));
      sizes.push_back(builder.getIndexAttr(info.numElements));
      strides.push_back(builder.getIndexAttr(1));
    } else {
      return allocOp.emitOpError()
             << "unsupported scratch shape rank: " << scratchShape.size();
    }

    // Create subview. For 1D result from [1, N] source, this performs
    // rank-reduction by dropping the unit dimension.
    Value result;
    if (resultShape.size() == 1 && scratchShape.size() == 2) {
      // Rank-reducing subview: [1, N] -> [M]
      // The result type drops the first dimension (size 1).
      auto subviewOp = builder.create<memref::SubViewOp>(
          loc, resultType, scratchMemRef, offsets, sizes, strides);
      result = subviewOp.getResult();
    } else if (resultShape.size() == 2 && scratchShape.size() == 2 &&
               resultShape[0] == 1) {
      // Same rank: [1, N] -> [1, M]
      auto subviewOp = builder.create<memref::SubViewOp>(
          loc, resultType, scratchMemRef, offsets, sizes, strides);
      result = subviewOp.getResult();
    } else if (resultShape.size() == 1 && scratchShape.size() == 1) {
      // 1D to 1D: [N] -> [M]
      auto subviewOp = builder.create<memref::SubViewOp>(
          loc, resultType, scratchMemRef, offsets, sizes, strides);
      result = subviewOp.getResult();
    } else {
      // For other shapes, create subview with inferred type first, then cast
      auto subviewOp = builder.create<memref::SubViewOp>(
          loc, scratchMemRef, offsets, sizes, strides);
      result = subviewOp.getResult();

      if (result.getType() != resultType) {
        // Try memref.cast for compatible layouts
        if (memref::CastOp::areCastCompatible(result.getType(), resultType)) {
          result = builder.create<memref::CastOp>(loc, resultType, result);
        } else {
          return allocOp.emitOpError()
                 << "cannot create subview with result type " << resultType
                 << " from scratch type " << scratchMemRefType
                 << ". Consider using a 1D scratch allocation shape.";
        }
      }
    }

    allocOp.getResult().replaceAllUsesWith(result);
    allocOp.erase();

    return success();
  }
};

} // namespace
