// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MLOWERSCRATCHALLOCATE
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
} // namespace mlir::tt::d2m

namespace {

/// Information about a single scratch allocation.
struct ScratchAllocationInfo {
  mlir::tt::d2m::ScratchAllocateOp op;
  int64_t slotId;
  int64_t sizeBytes;
  int64_t numElements;      // Total number of elements (tiles) in the memref
  int64_t assignedOffset;   // Offset in the master scratchpad (in elements)
  mlir::Type elemType;      // Element type (e.g., !ttcore.tile<32x32xbf16>)
  mlir::Attribute memSpace; // Memory space attribute
};

/// Get the total number of elements in a memref.
static int64_t getMemRefNumElements(mlir::MemRefType memrefType) {
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim == mlir::ShapedType::kDynamic) {
      return -1;
    }
    numElements *= dim;
  }
  return numElements;
}

/// Get the total size in bytes for a memref type.
static int64_t getMemRefSizeBytes(mlir::MemRefType memrefType) {
  mlir::Type elemType = memrefType.getElementType();
  int64_t numElements = getMemRefNumElements(memrefType);
  if (numElements < 0) {
    return -1;
  }

  // Use ttcore::getElementSizeBytes for proper handling of TileType
  uint64_t elemSizeBytes = mlir::tt::ttcore::getElementSizeBytes(elemType);
  return numElements * static_cast<int64_t>(elemSizeBytes);
}

/// Key for grouping allocations by element type.
struct ElementTypeKey {
  mlir::Type elemType;
  mlir::Attribute memSpace;

  bool operator==(const ElementTypeKey &other) const {
    return elemType == other.elemType && memSpace == other.memSpace;
  }
};

} // namespace

namespace llvm {
template <>
struct DenseMapInfo<ElementTypeKey> {
  static ElementTypeKey getEmptyKey() {
    return {DenseMapInfo<mlir::Type>::getEmptyKey(),
            DenseMapInfo<mlir::Attribute>::getEmptyKey()};
  }
  static ElementTypeKey getTombstoneKey() {
    return {DenseMapInfo<mlir::Type>::getTombstoneKey(),
            DenseMapInfo<mlir::Attribute>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ElementTypeKey &key) {
    return llvm::hash_combine(
        DenseMapInfo<mlir::Type>::getHashValue(key.elemType),
        DenseMapInfo<mlir::Attribute>::getHashValue(key.memSpace));
  }
  static bool isEqual(const ElementTypeKey &lhs, const ElementTypeKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {

/// Information about a master scratchpad for a specific element type.
struct MasterScratchpadInfo {
  mlir::Type elemType;
  mlir::Attribute memSpace;
  int64_t totalElements;
  int64_t totalSizeBytes;
  mlir::Value allocOp; // The memref.alloc for this master scratchpad
};

class D2MLowerScratchAllocatePass
    : public mlir::tt::d2m::impl::D2MLowerScratchAllocateBase<
          D2MLowerScratchAllocatePass> {
public:
  using mlir::tt::d2m::impl::D2MLowerScratchAllocateBase<
      D2MLowerScratchAllocatePass>::D2MLowerScratchAllocateBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Get the chip descriptor for alignment info
    mlir::tt::ttcore::SystemDescAttr systemDesc =
        mlir::tt::ttcore::getCurrentScopeSystemDesc(moduleOp);
    mlir::tt::ttcore::ChipDescAttr chipDesc = systemDesc.getChipDescs().front();
    nocL1AlignBytes = chipDesc.getNocL1AddressAlignBytes();

    // Process each function
    mlir::WalkResult result = moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      if (failed(processFunction(funcOp))) {
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

private:
  int64_t nocL1AlignBytes = 16; // Default alignment, overwritten from chip desc

  mlir::LogicalResult processFunction(mlir::func::FuncOp funcOp) {
    // Step 1: Collect all scratch_allocate ops
    llvm::SmallVector<ScratchAllocationInfo> allocations;
    bool hasError = false;

    funcOp.walk([&](mlir::tt::d2m::ScratchAllocateOp op) {
      mlir::MemRefType memrefType =
          mlir::cast<mlir::MemRefType>(op.getResult().getType());

      ScratchAllocationInfo info;
      info.op = op;
      info.slotId = op.getSlot();
      info.sizeBytes = getMemRefSizeBytes(memrefType);
      info.numElements = getMemRefNumElements(memrefType);
      info.assignedOffset = 0;
      info.elemType = memrefType.getElementType();
      info.memSpace = memrefType.getMemorySpace();

      if (info.sizeBytes < 0 || info.numElements < 0) {
        op.emitError(
            "scratch allocation has unsupported type or dynamic shape");
        hasError = true;
        return;
      }

      allocations.push_back(info);
    });

    if (hasError) {
      return mlir::failure();
    }

    // If no scratch allocations, nothing to do
    if (allocations.empty()) {
      return mlir::success();
    }

    // Step 2: Group allocations by element type and compute offsets
    llvm::DenseMap<ElementTypeKey, llvm::SmallVector<ScratchAllocationInfo *>>
        allocationsByType;

    for (ScratchAllocationInfo &info : allocations) {
      ElementTypeKey key{info.elemType, info.memSpace};
      allocationsByType[key].push_back(&info);
    }

    // Step 3: Compute offsets within each type group and track totals
    llvm::DenseMap<ElementTypeKey, MasterScratchpadInfo> masterScratchpads;
    int64_t grandTotalSizeBytes = 0;

    for (auto &[key, typeAllocations] : allocationsByType) {
      MasterScratchpadInfo &masterInfo = masterScratchpads[key];
      masterInfo.elemType = key.elemType;
      masterInfo.memSpace = key.memSpace;
      masterInfo.totalElements = 0;
      masterInfo.totalSizeBytes = 0;

      for (ScratchAllocationInfo *info : typeAllocations) {
        info->assignedOffset = masterInfo.totalElements;
        masterInfo.totalElements += info->numElements;
        masterInfo.totalSizeBytes += info->sizeBytes;
      }

      grandTotalSizeBytes += masterInfo.totalSizeBytes;
    }

    // Step 4: Check against maximum allowed size
    if (grandTotalSizeBytes > maxScratchSizeBytes) {
      funcOp.emitError("total scratch requirement (")
          << grandTotalSizeBytes << " bytes) exceeds maximum allowed ("
          << maxScratchSizeBytes << " bytes)";
      return mlir::failure();
    }

    // Step 5: Insert master scratchpad allocations at function entry
    mlir::OpBuilder builder(funcOp.getContext());
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    mlir::Location loc = funcOp.getLoc();

    // Track cumulative offset for address assignment
    int64_t currentAddressOffset = 0;

    for (auto &[key, masterInfo] : masterScratchpads) {
      // Create the master scratchpad memref type (1D with total elements)
      mlir::MemRefType masterType =
          mlir::MemRefType::get({masterInfo.totalElements}, masterInfo.elemType,
                                nullptr, masterInfo.memSpace);

      auto masterAlloc = builder.create<mlir::memref::AllocOp>(loc, masterType);

      // Set address and alignment attributes (required for D2MToTTMetal
      // lowering)
      int64_t address = scratchBaseAddress + currentAddressOffset;
      masterAlloc.setAlignment(nocL1AlignBytes);
      masterAlloc->setAttr("address", builder.getI64IntegerAttr(address));

      // Advance offset for next master scratchpad (align to NOC alignment)
      currentAddressOffset += masterInfo.totalSizeBytes;
      // Round up to alignment boundary
      currentAddressOffset =
          ((currentAddressOffset + nocL1AlignBytes - 1) / nocL1AlignBytes) *
          nocL1AlignBytes;

      masterInfo.allocOp = masterAlloc;
    }

    // Step 6: Replace each scratch_allocate with a subview + reinterpret_cast
    for (ScratchAllocationInfo &info : allocations) {
      builder.setInsertionPoint(info.op);

      ElementTypeKey key{info.elemType, info.memSpace};
      mlir::Value masterAlloc = masterScratchpads[key].allocOp;

      mlir::MemRefType resultType =
          mlir::cast<mlir::MemRefType>(info.op.getResult().getType());
      llvm::ArrayRef<int64_t> resultShape = resultType.getShape();

      // Create subview: extract a 1D slice from the master scratchpad
      llvm::SmallVector<mlir::OpFoldResult> offsets = {
          builder.getIndexAttr(info.assignedOffset)};
      llvm::SmallVector<mlir::OpFoldResult> sizes = {
          builder.getIndexAttr(info.numElements)};
      llvm::SmallVector<mlir::OpFoldResult> strides = {builder.getIndexAttr(1)};

      auto subview = builder.create<mlir::memref::SubViewOp>(
          info.op.getLoc(), masterAlloc, offsets, sizes, strides);

      mlir::Value replacement = subview;

      // Use reinterpret_cast to get the exact result type we need.
      // This handles both reshaping (1D -> nD) and removing the strided layout
      // that subview introduces. reinterpret_cast is more flexible than
      // expand_shape/cast for this purpose.
      mlir::MemRefType subviewType =
          mlir::cast<mlir::MemRefType>(subview.getType());
      if (subviewType != resultType) {
        // Compute strides for the result type (row-major order)
        llvm::SmallVector<int64_t> resultStrides(resultShape.size());
        int64_t stride = 1;
        for (int64_t i = resultShape.size() - 1; i >= 0; --i) {
          resultStrides[i] = stride;
          stride *= resultShape[i];
        }

        // Build sizes as OpFoldResult
        llvm::SmallVector<mlir::OpFoldResult> resultSizes;
        for (int64_t dim : resultShape) {
          resultSizes.push_back(builder.getIndexAttr(dim));
        }

        // Build strides as OpFoldResult
        llvm::SmallVector<mlir::OpFoldResult> resultStridesOFR;
        for (int64_t s : resultStrides) {
          resultStridesOFR.push_back(builder.getIndexAttr(s));
        }

        // Offset is 0 relative to the subview (we're at the start of our slice)
        mlir::OpFoldResult zeroOffset = builder.getIndexAttr(0);

        replacement = builder.create<mlir::memref::ReinterpretCastOp>(
            info.op.getLoc(), resultType, subview, zeroOffset, resultSizes,
            resultStridesOFR);
      }

      info.op.replaceAllUsesWith(replacement);
      info.op.erase();
    }

    // Step 7: Insert deallocs at function exit for all master scratchpads
    funcOp.walk([&](mlir::func::ReturnOp returnOp) {
      builder.setInsertionPoint(returnOp);
      for (auto &[key, masterInfo] : masterScratchpads) {
        builder.create<mlir::memref::DeallocOp>(loc, masterInfo.allocOp);
      }
    });

    return mlir::success();
  }
};

} // namespace
