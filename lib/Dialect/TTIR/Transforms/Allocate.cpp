// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLForwardCompat.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRALLOCATE
#define GEN_PASS_DEF_TTIRPLACEHOLDERALLOCATE
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper methods
//===----------------------------------------------------------------------===//

inline MemorySpace getMemorySpace(MemRefType memref) {
  return mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue();
}

inline MemorySpace getMemorySpace(MetalLayoutAttr layout) {
  return getMemorySpace(layout.getMemref());
}

inline MemorySpace getMemorySpace(RankedTensorType ty) {
  assert(ty.getEncoding());
  auto layout = mlir::cast<MetalLayoutAttr>(ty.getEncoding());
  return getMemorySpace(layout);
}

//===----------------------------------------------------------------------===//
// Allocate pass
//===----------------------------------------------------------------------===//

namespace {
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

      auto index = llvm::to_underlying(memorySpace);
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

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SystemDescAttr systemDesc = getCurrentScopeSystemDesc(module);
    ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    module->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      auto systemDesc = getCurrentScopeSystemDesc(func);
      assert(systemDesc);
      auto device = lookupDevice(func);
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

      func->walk([&](ttir::EmptyOp empty) {
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
} // namespace

namespace {
struct TTIRGenericFormStreams : public OpRewritePattern<ttir::GenericOp> {
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  static bool needsStream(AffineMap operandIndexingMap, ArrayAttr iteratorTypes,
                          Value operand, bool isOutput) {
    auto *definingOp = operand.getDefiningOp();
    if (mlir::isa_and_nonnull<ttir::StreamLayoutOp, memref::AllocOp>(
            definingOp)) {
      return false;
    }

    if (mlir::isa_and_nonnull<ttir::ViewLayoutOp>(definingOp)) {
      return true;
    }

    if (isOutput) {
      return false;
    }

    bool operandNeedsReduction = llvm::any_of(
        llvm::seq(operandIndexingMap.getNumResults()),
        [&](unsigned resultIndex) {
          unsigned dimPosition = operandIndexingMap.getDimPosition(resultIndex);
          IteratorType iteratorType =
              mlir::cast<IteratorTypeAttr>(iteratorTypes[dimPosition])
                  .getValue();
          return iteratorType == IteratorType::Reduction;
        });
    return operandNeedsReduction;
  }

  static void insertStream(PatternRewriter &rewriter, OpOperand &operand,
                           ttir::GenericOp op) {
    auto memref = mlir::cast<MemRefType>(operand.get().getType());
    auto streamAttr = rewriter.getAttr<StreamLayoutAttr>(
        rewriter.getMultiDimIdentityMap(memref.getRank()));
    auto streamMemref =
        MemRefType::get(memref.getShape(), memref.getElementType(), streamAttr,
                        memref.getMemorySpace());
    auto storage = rewriter.create<memref::AllocOp>(op.getLoc(), memref);
    auto streamLayout = rewriter.create<ttir::StreamLayoutOp>(
        op.getLoc(), streamMemref, operand.get(), storage);
    rewriter.modifyOpInPlace(
        op, [&]() { operand.assign(streamLayout.getResult()); });
  }

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

      if (!needsStream(operandIndexingMap, iteratorTypes, operand.get(),
                       isOutput)) {
        continue;
      }

      insertStream(rewriter, operand, op);

      modified = true;
    }

    return modified ? success() : failure();
  }
};
} // namespace

namespace {
class TTIRPlaceholderAllocate
    : public impl::TTIRPlaceholderAllocateBase<TTIRPlaceholderAllocate> {

  using impl::TTIRPlaceholderAllocateBase<
      TTIRPlaceholderAllocate>::TTIRPlaceholderAllocateBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericFormStreams>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
