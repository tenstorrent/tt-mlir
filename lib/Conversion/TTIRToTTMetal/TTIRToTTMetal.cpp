// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tt::ttmetal {

namespace {
class TTIRGenericRewriter : public OpConversionPattern<ttir::GenericOp> {
public:
  using OpConversionPattern<ttir::GenericOp>::OpConversionPattern;

  static ttmetal::KernelArgAttr evalCompileTimeArg(Builder &builder,
                                                   ArrayRef<Value> buffers,
                                                   ttkernel::ArgAttr arg) {
    switch (arg.getArgType()) {
    case ttkernel::ArgType::CBPort: {
      return builder.getAttr<ttmetal::KernelArgAttr>(
          arg.getOperandIndex(), arg.getArgType(), arg.getOperandIndex());
    }
    case ttkernel::ArgType::BufferAddress: {
      auto createBufferOp =
          buffers[arg.getOperandIndex()].getDefiningOp<CreateBufferOp>();
      assert(createBufferOp && "expected create buffer op");
      return builder.getAttr<ttmetal::KernelArgAttr>(
          arg.getOperandIndex(), arg.getArgType(), createBufferOp.getAddress());
    }
    case ttkernel::ArgType::Semaphore: {
      assert(false && "not implemented");
      break;
    }
    }
  }

  static ttmetal::KernelArgAttr
  evalRuntimeTimeArg(Builder &builder, ttkernel::ArgAttr arg) {
    return builder.getAttr<ttmetal::KernelArgAttr>(arg.getOperandIndex(),
                                                   arg.getArgType());
  }

  static KernelArgsAttr evalKernelArgsFromSpec(Builder &builder,
                                               ArrayRef<Value> buffers,
                                               SymbolTable const &symbolTable,
                                               SymbolRefAttr kernelSymbol) {
    auto kernelFunc =
        symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
    ttkernel::ArgSpecAttr kernelSpec =
        kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
            ttkernel::ArgSpecAttr::name);
    SmallVector<ttmetal::KernelArgAttr> rtArgs;
    SmallVector<ttmetal::KernelArgAttr> ctArgs;
    for (ttkernel::ArgAttr kernelArg : kernelSpec.getRtArgs()) {
      rtArgs.push_back(evalRuntimeTimeArg(builder, kernelArg));
    }
    for (ttkernel::ArgAttr kernelArg : kernelSpec.getCtArgs()) {
      ctArgs.push_back(evalCompileTimeArg(builder, buffers, kernelArg));
    }
    return builder.getAttr<ttmetal::KernelArgsAttr>(rtArgs, ctArgs);
  }

  static ArrayAttr
  convertThreadsToKernelConfigs(Builder &builder, ArrayAttr threads,
                                GridAttr opGrid, ArrayRef<Value> buffers,
                                SymbolTable const &symbolTable) {
    SmallVector<Attribute> kernelConfigs;
    uint32_t nocIndex = 0;
    auto coreRange = builder.getAttr<ttmetal::CoreRangeAttr>(opGrid);
    for (Attribute threadAttr : threads) {
      ttir::ThreadAttr thread = mlir::cast<ttir::ThreadAttr>(threadAttr);
      KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
          builder, buffers, symbolTable, thread.getKernelSymbol());
      Attribute kernelConfig = nullptr;
      switch (thread.getThreadType()) {
      case ttir::ThreadType::Compute: {
        kernelConfig = builder.getAttr<ttmetal::ComputeConfigAttr>(
            thread.getKernelSymbol(), coreRange, kernelArgs);
        break;
      }
      case ttir::ThreadType::Datamovement: {
        // The following assert just does a simple check for now, but in the
        // future you could have non-overlapping grids which would make this
        // calculation invalid.
        assert(nocIndex < 2 );
        kernelConfig = builder.getAttr<ttmetal::NocConfigAttr>(
            thread.getKernelSymbol(), coreRange, kernelArgs,
            *symbolizeNocIndex(nocIndex));
        ++nocIndex;
        break;
      }
      }
      assert(kernelConfig != nullptr);
      kernelConfigs.push_back(kernelConfig);
    }
    return builder.getArrayAttr(kernelConfigs);
  }

  LogicalResult
  matchAndRewrite(ttir::GenericOp op, ttir::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Value> buffers;
    llvm::SmallVector<Value> remappedBuffers;
    llvm::SmallVector<Value> cbs;
    for (auto operand : adaptor.getOperands()) {
      auto stream = mlir::dyn_cast_if_present<ttir::StreamLayoutOp>(
          operand.getDefiningOp());
      if (stream) {
        buffers.push_back(stream.getInput());
        remappedBuffers.push_back(rewriter.getRemappedValue(stream.getInput()));
        cbs.push_back(stream.getStorage());
      } else {
        buffers.push_back(operand);
        remappedBuffers.push_back(rewriter.getRemappedValue(operand));
        cbs.push_back(operand);
      }
    }

    ArrayAttr threads = op.getThreads();
    GridAttr opGrid = op.getGrid();
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto kernelConfigs = convertThreadsToKernelConfigs(
        rewriter, threads, opGrid, remappedBuffers, symbolTable);
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(op, buffers, cbs,
                                                           kernelConfigs);
    return success();
  };
};
} // namespace

namespace {
class MemrefAllocRewriter : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, memref::AllocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto device = lookupDevice(op);
    auto address = op->getAttr("address")
                       ? op->getAttrOfType<IntegerAttr>("address")
                       : rewriter.getI64IntegerAttr(
                             1000); // TODO(#1909): arbitrary default for now,
                                    // remove when allocate pass is implemented
    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memroy space found, failing.");
    auto memrefType = op.getMemref().getType();
    auto size = device.getMemrefSizeBytes(memrefType, 0);
    auto memorySpace =
      mlir::cast<tt::MemorySpaceAttr>(memrefType.getMemorySpace());
    auto createBufferOp = rewriter.create<ttmetal::CreateBufferOp>(
        op->getLoc(), memrefType, address.getInt(), size,
        memorySpace.getValue());
    rewriter.replaceOp(op, createBufferOp);

    return success();
  };
};
} // namespace

namespace {
class TTIRToLayoutRewriter : public OpConversionPattern<ttir::ToLayoutOp> {
public:
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, ttir::ToLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input = op.getInput();
    Value output = op.getOutput();
    MemRefType inputTy = mlir::cast<MemRefType>(input.getType());
    MemRefType outputTy = mlir::cast<MemRefType>(output.getType());
    tt::MemorySpaceAttr inputMemorySpace =
        mlir::dyn_cast_if_present<tt::MemorySpaceAttr>(
            inputTy.getMemorySpace());
    tt::MemorySpaceAttr outputMemorySpace =
        mlir::dyn_cast_if_present<tt::MemorySpaceAttr>(
            outputTy.getMemorySpace());
    bool inputMemorySpaceSet = inputMemorySpace != nullptr;
    bool outputMemorySpaceSet = outputMemorySpace != nullptr;
    assert((inputMemorySpaceSet != outputMemorySpaceSet) &&
           "expected either input or output to have memory space");

    // No memoryspace implicitly means host
    if (inputMemorySpace) {
      rewriter.replaceOpWithNewOp<ttmetal::EnqueueReadBufferOp>(op, input,
                                                                output);
    } else {
      rewriter.replaceOpWithNewOp<ttmetal::EnqueueWriteBufferOp>(op, input,
                                                                 output);
    }
    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateTTIRToTTMetalPatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns,
                                   TypeConverter & /*typeConverter*/) {
  patterns.add<ttmetal::TTIRGenericRewriter, ttmetal::MemrefAllocRewriter,
               ttmetal::TTIRToLayoutRewriter>(ctx);
}

} // namespace mlir::tt
