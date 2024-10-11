// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

inline MemorySpace getMemorySpace(MemRefType memref) {
  return mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue();
}

inline MemorySpace getMemorySpace(LayoutAttr layout) {
  return getMemorySpace(layout.getMemref());
}

inline MemorySpace getMemorySpace(RankedTensorType ty) {
  assert(ty.getEncoding());
  auto layout = mlir::cast<LayoutAttr>(ty.getEncoding());
  return getMemorySpace(layout);
}

//===----------------------------------------------------------------------===//
// Allocate pass
//===----------------------------------------------------------------------===//

class TTIRAllocate : public impl::TTIRAllocateBase<TTIRAllocate> {
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
        : memorySpaceInfo(memorySpaceInfo) {
      currPtr.reserve(memorySpaceInfo.size());
      for (auto const &info : memorySpaceInfo) {
        currPtr.push_back(info.baseAddress);
      }
    }

    uint64_t allocate(uint64_t size, MemorySpace memorySpace) {
      if (isSystemMemorySpace(memorySpace)) {
        return 0;
      }

      auto index = ttmlir::utils::enum_as_int(memorySpace);
      uint64_t &ptr = currPtr[index];
      ptr = ttmlir::utils::alignUp(ptr, memorySpaceInfo[index].alignment);
      auto result = ptr;
      ptr += size;
      assert(ptr <= memorySpaceInfo[index].end() && "Out of memory");
      return result;
    }

    SmallVector<uint64_t> currPtr;
    SmallVector<MemorySpaceInfo> memorySpaceInfo;
  };

public:
  using impl::TTIRAllocateBase<TTIRAllocate>::TTIRAllocateBase;

  std::pair<Operation *, Operation *>
  getStartEndOperationThroughDPSOps(const LivenessBlockInfo *livenessInfo,
                                    Value value) {
    auto *startOp = livenessInfo->getStartOperation(value);
    auto *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });
    assert(opOperandIter != endOp->getOpOperands().end());
    while (
        isa<DestinationStyleOpInterface>(endOp) and
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      assert(endOp->getResults().size() == 1);
      auto result = endOp->getResult(0);
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
      assert(opOperandIter != endOp->getOpOperands().end());
    }
    return std::make_pair(startOp, endOp);
  }

  SimpleAllocator createSimpleAllocator(ChipDescAttr chipDesc) {
    SmallVector<SimpleAllocator::MemorySpaceInfo> memorySpaceInfo;
    memorySpaceInfo.resize(getMaxEnumValForMemorySpace() + 1llu);
    memorySpaceInfo[ttmlir::utils::enum_as_int(MemorySpace::DeviceL1)] =
        SimpleAllocator::MemorySpaceInfo(chipDesc.getL1UnreservedBase(),
                                         chipDesc.getL1Size(),
                                         chipDesc.getNocL1AddressAlignBytes());
    memorySpaceInfo[ttmlir::utils::enum_as_int(MemorySpace::DeviceDRAM)] =
        SimpleAllocator::MemorySpaceInfo(
            chipDesc.getDramUnreservedBase(), chipDesc.getDramChannelSize(),
            chipDesc.getNocDRAMAddressAlignBytes());
    return SimpleAllocator(memorySpaceInfo);
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SystemDescAttr systemDesc = getCurrentScopeSystemDesc(module);
    ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    module->walk([&](func::FuncOp func) {
      assert(func.getBody().hasOneBlock());
      auto systemDesc = getCurrentScopeSystemDesc(func);
      assert(systemDesc);
      auto device = getCurrentScopeDevice(func);
      assert(device);
      SimpleAllocator allocator = createSimpleAllocator(chipDesc);
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      mlir::SmallVector<Attribute> argumentAllocations;
      for (auto operand : func.getArguments()) {
        auto operandTy = mlir::cast<RankedTensorType>(operand.getType());
        assert(operandTy.getEncoding());
        auto memorySpace = getMemorySpace(operandTy);
        auto sizeBytes = device.getTensorSizeBytes(operandTy, memorySpace);
        auto address = allocator.allocate(sizeBytes, memorySpace);
        argumentAllocations.push_back(rewriter.getAttr<ArgumentAllocationAttr>(
            address, sizeBytes, memorySpace));
      }
      func->setDiscardableAttr(ArgumentAllocationAttr::name,
                               rewriter.getArrayAttr(argumentAllocations));

      func->walk([&](tensor::EmptyOp empty) {
        auto resultTy =
            mlir::cast<RankedTensorType>(empty.getResult().getType());
        assert(resultTy.getEncoding());

        auto [startOp, endOp] =
            getStartEndOperationThroughDPSOps(livenessInfo, empty.getResult());

        // Replace empty with allocate
        auto memorySpace = getMemorySpace(resultTy);
        auto sizeBytes = device.getTensorSizeBytes(resultTy, memorySpace);
        auto address = allocator.allocate(sizeBytes, memorySpace);
        rewriter.setInsertionPoint(startOp);
        auto alloc = rewriter.create<AllocOp>(startOp->getLoc(), resultTy,
                                              address, sizeBytes, memorySpace);
        rewriter.replaceOp(empty, alloc);

        // Insert deallocate unless this value is being returned
        if (isa<func::ReturnOp>(endOp)) {
          return;
        }
        rewriter.setInsertionPointAfter(endOp);
        rewriter.create<DeallocOp>(endOp->getLoc(), alloc.getResult());
      });
    });
  }
};
} // namespace mlir::tt::ttir
