// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

namespace {

/// Information about a single scratch allocation.
struct ScratchAllocationInfo {
  mlir::tt::d2m::ScratchAllocateOp op;
  int64_t slotId;
  int64_t sizeBytes;
  int64_t byteOffset; // Assigned byte offset in the scratch buffer
};

/// Get the total size in bytes for a memref type.
static int64_t getMemRefSizeBytes(mlir::MemRefType memrefType) {
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim == mlir::ShapedType::kDynamic) {
      return -1;
    }
    numElements *= dim;
  }
  uint64_t elemSizeBytes =
      mlir::tt::ttcore::getElementSizeBytes(memrefType.getElementType());
  return numElements * static_cast<int64_t>(elemSizeBytes);
}

class D2MLowerScratchAllocatePass
    : public mlir::tt::d2m::impl::D2MLowerScratchAllocateBase<
          D2MLowerScratchAllocatePass> {
public:
  using mlir::tt::d2m::impl::D2MLowerScratchAllocateBase<
      D2MLowerScratchAllocatePass>::D2MLowerScratchAllocateBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

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
  mlir::LogicalResult processFunction(mlir::func::FuncOp funcOp) {
    // Step 1: Find scratch_init op to get the scratch buffer
    mlir::tt::d2m::ScratchInitOp scratchInit = nullptr;
    funcOp.walk([&](mlir::tt::d2m::ScratchInitOp op) { scratchInit = op; });

    // Step 2: Collect all scratch_allocate ops
    llvm::SmallVector<ScratchAllocationInfo> allocations;
    bool hasError = false;

    funcOp.walk([&](mlir::tt::d2m::ScratchAllocateOp op) {
      auto memrefType = mlir::cast<mlir::MemRefType>(op.getResult().getType());
      int64_t sizeBytes = getMemRefSizeBytes(memrefType);

      if (sizeBytes < 0) {
        op.emitError("scratch allocation has dynamic shape");
        hasError = true;
        return;
      }

      allocations.push_back(
          {op, static_cast<int64_t>(op.getSlot()), sizeBytes, 0});
    });

    if (hasError) {
      return mlir::failure();
    }

    /* ---------------------------- */
    // If no scratch allocations but scratch_init exists and test mode is on,
    // insert a dummy scratch_allocate for testing purposes.
    // NOTE: We don't add a load here because that would be in the main function
    // rather than inside a kernel, and D2MToTTKernel patterns are designed for
    // kernel contexts. The real scratch usage will come from the FPU fission
    // pass.
    if (allocations.empty() && scratchInit && testInsertDummyScratch) {
      mlir::OpBuilder builder(funcOp.getContext());
      builder.setInsertionPointAfter(scratchInit);
      mlir::Location loc = scratchInit.getLoc();

      // Create a dummy 8-tile bf16 scratch allocation
      auto tileType = mlir::tt::ttcore::TileType::get(
          funcOp.getContext(), {32, 32}, mlir::tt::ttcore::DataType::BFloat16);
      auto memSpaceAttr = mlir::tt::ttcore::MemorySpaceAttr::get(
          funcOp.getContext(), mlir::tt::ttcore::MemorySpace::DeviceL1);
      auto scratchType = mlir::MemRefType::get(
          {8}, tileType, /*layout=*/nullptr, memSpaceAttr);

      auto dummyAlloc = builder.create<mlir::tt::d2m::ScratchAllocateOp>(
          loc, scratchType, /*slot=*/0);

      int64_t sizeBytes = getMemRefSizeBytes(scratchType);
      allocations.push_back({dummyAlloc, 0, sizeBytes, 0});
    }
    /* ---------------------------- */

    // If no scratch allocations, just erase scratch_init if present
    if (allocations.empty()) {
      if (scratchInit) {
        scratchInit.erase();
      }
      return mlir::success();
    }

    // Must have scratch_init if we have allocations
    if (!scratchInit) {
      funcOp.emitError("scratch_allocate ops found but no scratch_init");
      return mlir::failure();
    }

    mlir::Value scratchBuffer = scratchInit.getBuffer();
    auto bufferType = mlir::cast<mlir::MemRefType>(scratchBuffer.getType());
    int64_t bufferSizeBytes = bufferType.getShape()[0]; // 1D byte buffer

    // Step 3: Compute sequential byte offsets
    int64_t currentOffset = 0;
    for (ScratchAllocationInfo &info : allocations) {
      info.byteOffset = currentOffset;
      currentOffset += info.sizeBytes;
    }

    // Step 4: Validate total size fits in buffer
    if (currentOffset > bufferSizeBytes) {
      funcOp.emitError("total scratch requirement (")
          << currentOffset << " bytes) exceeds buffer size (" << bufferSizeBytes
          << " bytes)";
      return mlir::failure();
    }

    // Step 5: Replace each scratch_allocate with memref.view
    mlir::OpBuilder builder(funcOp.getContext());

    for (ScratchAllocationInfo &info : allocations) {
      builder.setInsertionPoint(info.op);
      mlir::Location loc = info.op.getLoc();

      // Create byte offset constant
      mlir::Value offset =
          builder.create<mlir::arith::ConstantIndexOp>(loc, info.byteOffset);

      // Get the target memref type from scratch_allocate result
      auto resultType =
          mlir::cast<mlir::MemRefType>(info.op.getResult().getType());

      // Create memref.view: byte_buffer[offset] -> typed memref
      // For static shapes, sizes array is empty
      auto view = builder.create<mlir::memref::ViewOp>(
          loc, resultType, scratchBuffer, offset,
          /*sizes=*/mlir::ValueRange{});

      info.op.replaceAllUsesWith(view.getResult());
      info.op.erase();
    }

    // Step 6: Erase scratch_init (consumed)
    scratchInit.erase();

    return mlir::success();
  }
};

} // namespace
