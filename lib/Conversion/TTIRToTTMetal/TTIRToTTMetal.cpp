// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTMetal/TTIRToTTMetal.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstdint>

namespace mlir::tt::ttmetal {

namespace {
class TTIRGenericRewriter : public OpConversionPattern<ttir::GenericOp> {
public:
  using OpConversionPattern<ttir::GenericOp>::OpConversionPattern;

  static KernelArgsAttr evalKernelArgsFromSpec(Builder &builder,
                                               const SymbolTable &symbolTable,
                                               SymbolRefAttr kernelSymbol) {
    auto kernelFunc =
        symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
    ttkernel::ArgSpecAttr kernelSpec =
        kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
            ttkernel::ArgSpecAttr::name);
    SmallVector<ttmetal::KernelArgAttr> rtArgs;
    SmallVector<ttmetal::KernelArgAttr> ctArgs;
    for (ttkernel::ArgAttr arg : kernelSpec.getRtArgs()) {
      rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
          arg.getArgType(), arg.getOperandIndex()));
    }
    for (ttkernel::ArgAttr arg : kernelSpec.getCtArgs()) {
      ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
          arg.getArgType(), arg.getOperandIndex()));
    }
    return builder.getAttr<ttmetal::KernelArgsAttr>(rtArgs, ctArgs);
  }

  static ArrayAttr
  convertThreadsToKernelConfigs(Builder &builder, mlir::ValueRange operands,
                                ArrayAttr threads, ttcore::GridAttr opGrid,
                                const SymbolTable &symbolTable) {
    SmallVector<Attribute> kernelConfigs;
    uint32_t nocIndex = 0;
    auto coreRange = builder.getAttr<ttmetal::CoreRangeAttr>(opGrid);
    for (Attribute threadAttr : threads) {
      ttir::ThreadAttr thread = mlir::cast<ttir::ThreadAttr>(threadAttr);
      KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
          builder, symbolTable, thread.getKernelSymbol());
      Attribute kernelConfig = nullptr;
      switch (thread.getThreadType()) {
      case ttir::ThreadType::Compute: {
        // TODO (wenbinlyuTT): enable f32 accum & unpack mode
        constexpr bool fp32DestAccum = false;
        std::vector<UnpackToDestMode> unpackModes{UnpackToDestMode::Default};
        kernelConfig = builder.getAttr<ttmetal::ComputeConfigAttr>(
            thread.getKernelSymbol(), coreRange, kernelArgs, fp32DestAccum,
            unpackModes);
        break;
      }
      case ttir::ThreadType::Datamovement: {
        // The following assert just does a simple check for now, but in the
        // future you could have non-overlapping grids which would make this
        // calculation invalid.
        assert(nocIndex < 2);
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
    llvm::SmallVector<int64_t> cbPorts;
    int64_t cbPort = 0;
    for (auto operand : adaptor.getOperands()) {
      if (auto stream = mlir::dyn_cast_if_present<ttir::StreamLayoutOp>(
              operand.getDefiningOp());
          stream) {
        buffers.push_back(stream.getInput());
        remappedBuffers.push_back(rewriter.getRemappedValue(stream.getInput()));
        cbs.push_back(stream.getStorage());
      } else if (auto view = mlir::dyn_cast_if_present<ttir::ViewLayoutOp>(
                     operand.getDefiningOp());
                 view) {
        buffers.push_back(view.getInput());
        remappedBuffers.push_back(rewriter.getRemappedValue(view.getInput()));
        cbs.push_back(view.getInput());
      } else {
        buffers.push_back(operand);
        remappedBuffers.push_back(rewriter.getRemappedValue(operand));
        cbs.push_back(operand);
      }
      cbPorts.push_back(cbPort++);
    }

    ArrayAttr threads = op.getThreads();
    ttcore::GridAttr opGrid = op.getGrid();
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto kernelConfigs = convertThreadsToKernelConfigs(
        rewriter, adaptor.getOperands(), threads, opGrid, symbolTable);
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, buffers, cbs, cbPorts, kernelConfigs);
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
    auto address = op->getAttrOfType<IntegerAttr>("address");
    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memory space found, failing.");
    auto memrefType = op.getMemref().getType();
    assert(mlir::isa<ttcore::ShardLayoutAttr>(memrefType.getLayout()));
    rewriter.replaceOpWithNewOp<ttmetal::CreateBufferOp>(op, memrefType,
                                                         address);

    return success();
  };
};
} // namespace

namespace {
class MemrefDeallocRewriter : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, memref::DeallocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::DeallocateBufferOp>(op,
                                                             op.getMemref());

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
    Value input;
    if (auto view = mlir::dyn_cast_if_present<ttir::ViewLayoutOp>(
            adaptor.getInput().getDefiningOp())) {
      input = view.getInput();
    } else {
      input = adaptor.getInput();
    }
    Value output;
    if (auto view = mlir::dyn_cast_if_present<ttir::ViewLayoutOp>(
            adaptor.getOutput().getDefiningOp())) {
      output = view.getInput();
    } else {
      output = adaptor.getOutput();
    }
    MemRefType inputTy = mlir::cast<MemRefType>(input.getType());
    MemRefType outputTy = mlir::cast<MemRefType>(output.getType());
    ttcore::MemorySpaceAttr inputMemorySpace =
        mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
            inputTy.getMemorySpace());
    ttcore::MemorySpaceAttr outputMemorySpace =
        mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
            outputTy.getMemorySpace());
    bool inputMemorySpaceSet = inputMemorySpace != nullptr;
    bool outputMemorySpaceSet = outputMemorySpace != nullptr;
    assert((inputMemorySpaceSet != outputMemorySpaceSet) &&
           "expected either input or output to have memory space");

    // No memoryspace implicitly means host.
    if (inputMemorySpace) {
      assert(!mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
          inputTy.getLayout()));
      rewriter.replaceOpWithNewOp<ttmetal::EnqueueReadBufferOp>(op, input,
                                                                output);
      // Insert global barrier to ensure the read completes before subsequent
      // ops use it.
      rewriter.create<ttmetal::FinishOp>(op->getLoc());
    } else {
      assert(!mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
          outputTy.getLayout()));
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
               ttmetal::MemrefDeallocRewriter, ttmetal::TTIRToLayoutRewriter>(
      ctx);
}

} // namespace mlir::tt
