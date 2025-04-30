// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallSet.h"

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

inline MemorySpace getMemorySpace(MemRefType memref, MemorySpace dflt) {
  auto memSpace = memref.getMemorySpace();
  return memSpace
             ? mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue()
             : dflt;
}

//===----------------------------------------------------------------------===//
// Helper classes.
//===----------------------------------------------------------------------===//
namespace {
struct SimpleAllocator {
  struct MemorySpaceInfo {
    uint64_t baseAddress = 0;
    uint64_t size = 0;
    uint64_t alignment = 0;

    MemorySpaceInfo() = default;
    MemorySpaceInfo(uint64_t baseAddress, uint64_t size, uint64_t alignment)
        : baseAddress(baseAddress), size(size), alignment(alignment) {}
    inline uint64_t end() const { return baseAddress + size; }
  };

  SimpleAllocator(SmallVector<MemorySpaceInfo> memorySpaceInfo)
      : currPtr(llvm::map_to_vector(
            memorySpaceInfo, [](auto &&info) { return info.baseAddress; })),
        memorySpaceInfo(memorySpaceInfo) {}

  std::array<uint64_t, 2> allocate(uint64_t size, MemorySpace memorySpace) {
    if (isSystemMemorySpace(memorySpace)) {
      return {0, 0};
    }

    const auto index = llvm::to_underlying(memorySpace);
    uint64_t &ptr = currPtr[index];
    const uint64_t alignment = memorySpaceInfo[index].alignment;
    ptr = ttmlir::utils::alignUp(ptr, alignment);
    const uint64_t result = ptr;
    ptr += size;
    assert(ptr <= memorySpaceInfo[index].end() && "Out of memory");
    return {result, alignment};
  }

  SmallVector<uint64_t> currPtr;
  SmallVector<MemorySpaceInfo> memorySpaceInfo;
}; // end of class
} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation.
//===----------------------------------------------------------------------===//
namespace {
class TTIRAllocateStreams final : public OpRewritePattern<ttir::GenericOp> {
  using base = OpRewritePattern<ttir::GenericOp>;

  using base::base;

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    ArrayAttr iteratorTypes = op.getIteratorTypes();
    for (OpOperand &operand : op->getOpOperands()) {
      bool isOutput = operand.getOperandNumber() >= outputOperandsIndex;
      AffineMap operandIndexingMap =
          mlir::cast<AffineMapAttr>(
              op.getIndexingMaps()[operand.getOperandNumber()])
              .getValue();

      if (!needsStream(operand.get(), isOutput, operandIndexingMap,
                       iteratorTypes)) {
        continue;
      }

      insertStream(rewriter, operand, op);
      modified = true;
    }

    return success(modified);
  }

  static bool needsStream(Value operand, bool isOutput,
                          AffineMap operandIndexingMap,
                          ArrayAttr iteratorTypes) {

    Operation *definingOp = operand.getDefiningOp();

    // Greedy driver fixed point reached?
    if (mlir::isa_and_nonnull<ttir::StreamLayoutOp>(definingOp)) {
      return false;
    }

    // A view_layout signals that an op wants to take a view of an
    // operand, possibly to switch to a different core grid shape.
    if (mlir::isa_and_nonnull<ttir::ViewLayoutOp>(definingOp)) {
      return true;
    }

    // No stream (NOC ops) will be needed if 'operand' is already
    // allocated in L1 ("alias" mode), which is currently guaranteed
    // to be the case for outputs.
    if (isOutput) {
      return false;
    }

    // A core participating in a reduction dim necessarily requires
    // non-local data movement unless it is the only core involved
    // in that dim.
    //
    // Similar logic applies to a broadcast dim.
    //
    // TODO(vroubtsov) we are currently fixing the core grid shape to be
    // equal to the output shape, hence could we not infer the *exact*
    // pattern of data movement that's not local to any core by walking
    // the operand/output affine maps?

    const auto bcastDims = operandIndexingMap.getBroadcastDims();
    const llvm::SmallSet<unsigned, 4> bcastDimIndex(bcastDims.begin(),
                                                    bcastDims.end());

    const bool operandNeedsDataMovement = llvm::any_of(
        llvm::seq(operandIndexingMap.getNumResults()),
        [&](unsigned resultIndex) {
          if (bcastDimIndex.contains(resultIndex)) {
            return true;
          }
          const auto dimPosition =
              operandIndexingMap.getDimPosition(resultIndex);
          IteratorType iteratorType =
              mlir::cast<IteratorTypeAttr>(iteratorTypes[dimPosition])
                  .getValue();
          return (iteratorType == IteratorType::Reduction);
        });
    return operandNeedsDataMovement;
  }

  static void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                           ttir::GenericOp op) {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());
    auto streamAttr = rewriter.getAttr<ViewLayoutAttr>(
        rewriter.getMultiDimIdentityMap(memref.getRank()));
    auto streamMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), streamAttr,
                        memref.getMemorySpace());
    auto storageAttr = ShardLayoutAttr::get(memref, /*buffers=*/1);
    auto storageMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), storageAttr,
                        memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), storageMemref);
    auto streamLayout = rewriter.create<ttir::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }

}; // end of class
} // namespace
// ............................................................................

namespace {
class TTIRAllocate final : public impl::TTIRAllocateBase<TTIRAllocate> {
  using base = impl::TTIRAllocateBase<TTIRAllocate>;

  using base::base;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Currently, these steps appear independent
    // this will change (e.g. to use shared analysis context).

    if (failed(runAllocateStreams(moduleOp))) {
      signalPassFailure();
      return;
    }

    if (failed(runAllocateBuffers(moduleOp))) {
      signalPassFailure();
      return;
    }
  }

  LogicalResult runAllocateStreams(ModuleOp moduleOp) {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRAllocateStreams>(&getContext());
    return mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
  }

  LogicalResult runAllocateBuffers(ModuleOp moduleOp) {

    SystemDescAttr systemDesc = getCurrentScopeSystemDesc(moduleOp);
    ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    auto result = moduleOp->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return WalkResult::skip();
      }

      if (failed(runAllocateBuffers(func, chipDesc))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    return success(!result.wasInterrupted());
  }

  LogicalResult runAllocateBuffers(func::FuncOp func,
                                   const ChipDescAttr &chipDesc) {
    assert(func.getBody().hasOneBlock() &&
           "found func that didn't have one block!");

    DeviceAttr device = lookupDevice(func);
    SimpleAllocator allocator = createSimpleAllocator(chipDesc);

    // Augment all 'memref.alloc's in device memory with allocated addresses and
    // correct alignments.

    IRRewriter rewriter(&getContext());

    func->walk([&](memref::AllocOp alloc) {
      MemRefType memrefTy = alloc.getType();
      MemorySpace memorySpace = getMemorySpace(
          memrefTy, MemorySpace::System); // Interpret unset as "host memory".

      if (!isDeviceMemorySpace(memorySpace)) {
        return;
      }

      const auto sizeBytes = device.getMemrefSizeBytes(memrefTy, 0);
      const auto [address, alignment] =
          allocator.allocate(sizeBytes, memorySpace);

      rewriter.startOpModification(alloc);
      {
        alloc.setAlignment(alignment);
        alloc->setAttr("address", rewriter.getI64IntegerAttr(address));
      };
      rewriter.finalizeOpModification(alloc);
    });

    // Currently, this step always succeeds; out-of-memory condition will result
    // in hard assert failure.

    return success();
  }

  static SimpleAllocator createSimpleAllocator(ChipDescAttr chipDesc) {
    SmallVector<SimpleAllocator::MemorySpaceInfo> memorySpaceInfo;
    memorySpaceInfo.resize(getMaxEnumValForMemorySpace() + 1llu);
    memorySpaceInfo[llvm::to_underlying(MemorySpace::DeviceL1)] =
        SimpleAllocator::MemorySpaceInfo(chipDesc.getL1UnreservedBase(),
                                         chipDesc.getL1Size() -
                                             chipDesc.getScratchL1RegionSize(),
                                         chipDesc.getNocL1AddressAlignBytes());
    memorySpaceInfo[llvm::to_underlying(MemorySpace::DeviceDRAM)] =
        SimpleAllocator::MemorySpaceInfo(
            chipDesc.getDramUnreservedBase(), chipDesc.getDramChannelSize(),
            chipDesc.getNocDRAMAddressAlignBytes());
    return SimpleAllocator(memorySpaceInfo);
  }

}; // end of class
} // namespace

} // namespace mlir::tt::ttir
// ----------------------------------------------------------------------------
