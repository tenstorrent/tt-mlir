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

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRALLOCATEBUFFERS
#define GEN_PASS_DEF_TTIRFORMSTREAMS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

inline MemorySpace getMemorySpace(MemRefType memref) {
  auto memSpace = memref.getMemorySpace();
  assert(memSpace && "expecting memrefs in explicit memory spaces");
  return mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue();
}

//===----------------------------------------------------------------------===//
// Allocate memref buffers pass.
//===----------------------------------------------------------------------===//
namespace {
// Visit all funcOps of the parent module and add "address" int64 attributes
// to all 'memref.alloc's computed by a simple bumper allocator.
class TTIRAllocateBuffers final
    : public impl::TTIRAllocateBuffersBase<TTIRAllocateBuffers> {

public:
  using impl::TTIRAllocateBuffersBase<
      TTIRAllocateBuffers>::TTIRAllocateBuffersBase;

private:
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
  }; // end of nested class

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    SystemDescAttr systemDesc = getCurrentScopeSystemDesc(moduleOp);
    ChipDescAttr chipDesc = systemDesc.getChipDescs().front();

    moduleOp->walk([&](func::FuncOp func) {
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

      // Augment 'memref.alloc's with allocated addresses.

      func->walk([&](memref::AllocOp alloc) {
        MemRefType memrefTy = alloc.getType();
        MemorySpace memorySpace = getMemorySpace(memrefTy);

        auto sizeBytes = device.getMemrefSizeBytes(memrefTy, 0);
        auto address = allocator.allocate(sizeBytes, memorySpace);
        rewriter.modifyOpInPlace(alloc, [&]() {
          alloc->setAttr("address", rewriter.getI64IntegerAttr(address));
        });
      });
    });
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

namespace {
// Visit ttir.generic ops and modify their operands to originate from
// ttir.stream_layouts where necessary.
class TTIRGenericFormStreams final : public OpRewritePattern<ttir::GenericOp> {

public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

private:
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

    // No stream (NOC ops) will be needed if 'operand' is already
    // allocated in L1 ("alias" mode), which is currently guaranteed
    // to be the case for outputs.
    if (isOutput || mlir::isa_and_nonnull<memref::AllocOp>(definingOp)) {
      return false;
    }

    // A core participating in a reduction dim necessarily requires
    // non-local data movement unless it is the only core involved
    // in that dim.
    //
    // TODO(vroubtsov) we are currently fixing the core grid shape to be
    // equal to the output shape, hence could we not infer the *exact*
    // pattern of data movement that's not local to any core by walking
    // the operand/output affine maps? that is, there seems to be no need
    // for "reduction always implies non-locality" heuristic?

    bool operandNeedsReduction = llvm::any_of(
        llvm::seq(operandIndexingMap.getNumResults()),
        [&](unsigned resultIndex) {
          unsigned dimPosition = operandIndexingMap.getDimPosition(resultIndex);
          IteratorType iteratorType =
              mlir::cast<IteratorTypeAttr>(iteratorTypes[dimPosition])
                  .getValue();
          return (iteratorType == IteratorType::Reduction);
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
}; // end of class
} // namespace

//===----------------------------------------------------------------------===//
// Form streams pass.
//===----------------------------------------------------------------------===//
namespace {
class TTIRFormStreams final
    : public impl::TTIRFormStreamsBase<TTIRFormStreams> {

  using impl::TTIRFormStreamsBase<TTIRFormStreams>::TTIRFormStreamsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericFormStreams>(&getContext());
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
}; // end of class
} // namespace

} // namespace mlir::tt::ttir
