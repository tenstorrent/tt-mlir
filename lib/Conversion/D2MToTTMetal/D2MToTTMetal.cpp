// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstdint>

namespace mlir::tt::ttmetal {

namespace {
static KernelArgsAttr evalKernelArgsFromSpec(Builder &builder,
                                             const SymbolTable &symbolTable,
                                             SymbolRefAttr kernelSymbol,
                                             uint32_t cbOffset) {
  auto kernelFunc =
      symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
  ttkernel::ArgSpecAttr kernelSpec =
      kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);
  SmallVector<ttmetal::KernelArgAttr> rtArgs;
  SmallVector<ttmetal::KernelArgAttr> ctArgs;
  for (ttkernel::ArgAttr arg : kernelSpec.getRtArgs()) {
    rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
        arg.getArgType(), arg.getOperandIndex() + cbOffset));
  }
  for (ttkernel::ArgAttr arg : kernelSpec.getCtArgs()) {
    ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
        arg.getArgType(), arg.getOperandIndex() + cbOffset));
  }
  return builder.getAttr<ttmetal::KernelArgsAttr>(rtArgs, ctArgs);
}

static ArrayAttr convertThreadsToKernelConfigs(
    Builder &builder, mlir::ValueRange operands, ArrayAttr threads,
    ttcore::GridAttr opGrid, const SymbolTable &symbolTable,
    ttmetal::MathFidelity mathFidelity, uint32_t cbOffset) {
  SmallVector<Attribute> kernelConfigs;
  uint32_t nocIndex = 0;

  auto coreRange = ttmetal::CoreRangeAttr::get(opGrid);

  for (Attribute threadAttr : threads) {
    d2m::ThreadAttr thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
    KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
        builder, symbolTable, thread.getKernelSymbol(), cbOffset);
    Attribute kernelConfig = nullptr;
    switch (thread.getThreadType()) {
    case d2m::ThreadType::Compute: {
      bool fp32DestAccum = false;
      for (size_t i = 0; i < operands.size(); ++i) {
        ttcore::DataType dataType = ttcore::elementTypeToDataType(
            ttcore::getOperandInnerElementType(operands[i]));

        if (getNumberOfBits(dataType) == 32) {
          fp32DestAccum = true;
        }
      }
      // This must stay in-sync with ChipDescAttr::getDstLogicalSizeTiles().
      constexpr bool dstFullSyncEn = false;
      std::vector<UnpackToDestMode> unpackModes{UnpackToDestMode::Default};
      kernelConfig = builder.getAttr<ttmetal::ComputeConfigAttr>(
          thread.getKernelSymbol(), coreRange, kernelArgs, mathFidelity,
          fp32DestAccum, dstFullSyncEn, unpackModes);
      break;
    }
    case d2m::ThreadType::Datamovement: {
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

static ttmetal::EnqueueProgramOp convertD2MGenericToEnqueueProgram(
    ConversionPatternRewriter &rewriter, d2m::GenericOp op, ValueRange operands,
    ttmetal::MathFidelity mathFidelity, uint32_t cbOffset = 0) {
  llvm::SmallVector<Value> buffers;
  llvm::SmallVector<Value> remappedBuffers;
  llvm::SmallVector<Value> cbs;
  llvm::SmallVector<int64_t> cbPorts;
  int64_t cbPort = 0;
  for (auto operand : operands) {
    if (auto stream = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
            operand.getDefiningOp());
        stream) {
      buffers.push_back(stream.getInput());
      remappedBuffers.push_back(rewriter.getRemappedValue(stream.getInput()));
      cbs.push_back(stream.getStorage());
    } else if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
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
      rewriter, operands, threads, opGrid, symbolTable, mathFidelity, cbOffset);

  return rewriter.create<ttmetal::EnqueueProgramOp>(op->getLoc(), buffers, cbs,
                                                    cbPorts, kernelConfigs);
}

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *ctx, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(ctx), mathFidelity_(mathFidelity) {}

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto enqueueProgramOp = convertD2MGenericToEnqueueProgram(
        rewriter, op, adaptor.getOperands(), mathFidelity_);
    rewriter.replaceOp(op, enqueueProgramOp);

    return success();
  };

private:
  ttmetal::MathFidelity mathFidelity_;
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

    auto layout = mlir::dyn_cast_if_present<ttcore::DeviceLayoutInterface>(
        memrefType.getLayout());
    assert(layout && layout.isPhysical() && "expected physical device layout");

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
class D2MToLayoutRewriter : public OpConversionPattern<d2m::ToLayoutOp> {
public:
  using OpConversionPattern<d2m::ToLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ToLayoutOp op, d2m::ToLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input;
    if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            adaptor.getInput().getDefiningOp())) {
      input = view.getInput();
    } else {
      input = adaptor.getInput();
    }
    Value output;
    if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
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

namespace {
class D2MMeshShardRewriter : public OpConversionPattern<d2m::MeshShardOp> {
public:
  using OpConversionPattern<d2m::MeshShardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::MeshShardOp op, d2m::MeshShardOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input = adaptor.getInput();

    // Use the op's result type directly because bufferization has already
    // determined the correct output shape. For shard_to_full operations, the
    // result shape differs from the input (e.g., concatenating shards), while
    // for full_to_shard it's the opposite. The bufferization pass correctly
    // handles these transformations.
    Type resultType = op.getResult().getType();

    rewriter.replaceOpWithNewOp<ttmetal::MeshShardOp>(
        op, resultType, input, op.getShardType(), op.getShardDirection(),
        op.getShardShape(), op.getShardDims());
    return success();
  };
};

} // namespace

namespace {
class D2MSpatialRewriter : public OpConversionPattern<d2m::SpatialOp> {
public:
  D2MSpatialRewriter(MLIRContext *ctx, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::SpatialOp>(ctx), mathFidelity_(mathFidelity) {}

  LogicalResult
  matchAndRewrite(d2m::SpatialOp op, d2m::SpatialOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto opOperands = op.getOperands();

    auto getValuesFromBlockArgs = [&](OperandRange args) -> SmallVector<Value> {
      SmallVector<Value> values;
      for (auto arg : args) {
        if (mlir::BlockArgument blockArg =
                mlir::dyn_cast_if_present<mlir::BlockArgument>(arg)) {
          values.push_back(opOperands[blockArg.getArgNumber()]);
        } else {
          llvm_unreachable("Expected BlockArgument.");
        }
      }
      return values;
    };

    ttmetal::PerDeviceProgramArgsAttr ttmetalPerDeviceProgramArgsAttr = nullptr;
    SmallVector<ttmetal::ProgramArgsAttr> ttmetalPerDeviceProgramArgs;
    if (auto perDeviceProgramArgsAttr = op.getPerDeviceProgramArgsAttr()) {
      for (auto programArgsAttr :
           perDeviceProgramArgsAttr.getPerDeviceProgramArgs()) {
        SmallVector<ttmetal::KernelArgsAttr> ttmetalKernelArgs;
        for (auto kernelArgsAttr : programArgsAttr.getProgramArgs()) {
          kernelArgsAttr.dump();
          SmallVector<ttmetal::KernelArgAttr> CtKernelArgs;
          SmallVector<ttmetal::KernelArgAttr> RtKernelArgs;
          for (auto ctArg : kernelArgsAttr.getCtArgs()) {
            CtKernelArgs.push_back(ttmetal::KernelArgAttr::get(
                rewriter.getContext(), ttkernel::ArgType::CBPort, ctArg));
          }
          for (auto rtArg : kernelArgsAttr.getRtArgs()) {
            RtKernelArgs.push_back(ttmetal::KernelArgAttr::get(
                rewriter.getContext(), ttkernel::ArgType::Num, rtArg));
          }
          ttmetalKernelArgs.push_back(ttmetal::KernelArgsAttr::get(
              rewriter.getContext(), RtKernelArgs, CtKernelArgs));
        }
        ttmetalPerDeviceProgramArgs.push_back(ttmetal::ProgramArgsAttr::get(
            rewriter.getContext(), ttmetalKernelArgs));
      }
      ttmetalPerDeviceProgramArgsAttr = ttmetal::PerDeviceProgramArgsAttr::get(
          rewriter.getContext(), ttmetalPerDeviceProgramArgs);
    }

    SmallVector<Value> concatenatedBuffers;
    SmallVector<Value> concatenatedCbs;
    SmallVector<int64_t> concatenatedCbPorts;
    SmallVector<Attribute> concatenatedKernelConfigs;
    uint32_t cbOffset = 0;
    for (auto &blockOp : op.getBody().getOps()) {
      if (auto d2mGenericOp = dyn_cast_if_present<d2m::GenericOp>(blockOp)) {
        auto enqueueProgramOp = convertD2MGenericToEnqueueProgram(
            rewriter, d2mGenericOp, d2mGenericOp.getOperands(), mathFidelity_,
            cbOffset);
        auto buffers = getValuesFromBlockArgs(enqueueProgramOp.getBuffers());
        auto cbs = getValuesFromBlockArgs(enqueueProgramOp.getCbs());
        auto cbPorts = enqueueProgramOp.getCbPorts();
        auto kernelConfigs = enqueueProgramOp.getKernelConfigs();
        concatenatedBuffers.append(buffers.begin(), buffers.end());
        concatenatedCbs.append(cbs.begin(), cbs.end());
        concatenatedCbPorts.append(cbPorts.begin(), cbPorts.end());
        concatenatedKernelConfigs.append(kernelConfigs.begin(),
                                         kernelConfigs.end());
        rewriter.eraseOp(enqueueProgramOp);
        cbOffset += cbPorts.size();
      } else {
        llvm_unreachable("Expected only d2m::GenericOp in spatial op body.");
      }
    }

    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, concatenatedBuffers, concatenatedCbs, concatenatedCbPorts,
        rewriter.getArrayAttr(concatenatedKernelConfigs),
        ttmetalPerDeviceProgramArgsAttr);

    return success();
  };

private:
  ttmetal::MathFidelity mathFidelity_;
};
} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateD2MToTTMetalPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter & /*typeConverter*/,
                                  ttmetal::MathFidelity mathFidelity) {
  patterns.add<ttmetal::MemrefAllocRewriter, ttmetal::MemrefDeallocRewriter,
               ttmetal::D2MToLayoutRewriter, ttmetal::D2MMeshShardRewriter>(
      ctx);
  patterns.add<ttmetal::D2MGenericRewriter, ttmetal::D2MSpatialRewriter>(
      ctx, mathFidelity);
}

} // namespace mlir::tt
