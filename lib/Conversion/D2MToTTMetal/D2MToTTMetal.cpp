// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <cstdint>

namespace mlir::tt::ttmetal {

namespace {

// Returns true if the kernel function contains an op of type OpT.
template <typename OpT>
static bool kernelContainsOp(const SymbolTable &symbolTable,
                             SymbolRefAttr kernelSymbol) {
  auto kernelFunc =
      symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
  return kernelFunc.walk([](OpT) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *ctx, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(ctx), mathFidelity_(mathFidelity) {}

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

  static ArrayAttr convertThreadsToKernelConfigs(
      Builder &builder, mlir::ValueRange operands, ArrayAttr threads,
      ArrayRef<int64_t> physicalGridShape, const SymbolTable &symbolTable,
      ttmetal::MathFidelity mathFidelity) {
    SmallVector<Attribute> kernelConfigs;
    uint32_t nocIndex = 0;

    auto coreRange = ttmetal::CoreRangeAttr::getPhysicalCoreRange(
        builder.getContext(), physicalGridShape);

    for (Attribute threadAttr : threads) {
      d2m::ThreadAttr thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
      KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
          builder, symbolTable, thread.getKernelSymbol());
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
        // Enable fp32 unpack mode for typecast kernels.
        // TODO(ckaravasilisTT): Enable fp32 unpack mode in the general case.
        bool isTypecast = kernelContainsOp<ttkernel::TypecastTileOp>(
            symbolTable, thread.getKernelSymbol());
        UnpackToDestMode mode = (fp32DestAccum && isTypecast)
                                    ? UnpackToDestMode::Fp32
                                    : UnpackToDestMode::Default;
        std::vector<UnpackToDestMode> unpackModes{mode};
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
      case d2m::ThreadType::Unified: {
        // Unified threads should have been split by SplitUnifiedThread before
        // reaching this pass.
        llvm_unreachable(
            "Unified threads are not supported for TTMetal backend");
      }
      }
      assert(kernelConfig != nullptr);
      kernelConfigs.push_back(kernelConfig);
    }
    return builder.getArrayAttr(kernelConfigs);
  }

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Value> buffers;
    llvm::SmallVector<Value> remappedBuffers;
    llvm::SmallVector<Value> cbs;
    llvm::SmallVector<int64_t> cbPorts;
    int64_t cbPort = 0;
    for (auto operand : adaptor.getOperands()) {
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
    auto physicalGridShape = op.getPhysicalGridShape();
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto kernelConfigs = convertThreadsToKernelConfigs(
        rewriter, adaptor.getOperands(), threads, physicalGridShape,
        symbolTable, mathFidelity_);
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, buffers, cbs, cbPorts, kernelConfigs, nullptr);
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

//===----------------------------------------------------------------------===//
// Host Transfer Ops: ToDeviceOp -> EnqueueWriteBuffer, ToHostOp ->
// EnqueueReadBuffer
//===----------------------------------------------------------------------===//

namespace {
class D2MToDeviceRewriter : public OpConversionPattern<d2m::ToDeviceOp> {
public:
  using OpConversionPattern<d2m::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ToDeviceOp op, d2m::ToDeviceOpAdaptor adaptor,
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

    [[maybe_unused]] MemRefType outputTy =
        mlir::cast<MemRefType>(output.getType());
    assert(!mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
               outputTy.getLayout()) &&
           "output should be device memory");

    rewriter.replaceOpWithNewOp<ttmetal::EnqueueWriteBufferOp>(op, input,
                                                               output);
    return success();
  }
};
} // namespace

namespace {
class D2MToHostRewriter : public OpConversionPattern<d2m::ToHostOp> {
public:
  using OpConversionPattern<d2m::ToHostOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ToHostOp op, d2m::ToHostOpAdaptor adaptor,
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

    [[maybe_unused]] MemRefType inputTy =
        mlir::cast<MemRefType>(input.getType());
    assert(!mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
               inputTy.getLayout()) &&
           "input should be device memory");

    rewriter.replaceOpWithNewOp<ttmetal::EnqueueReadBufferOp>(op, input,
                                                              output);
    // Insert global barrier to ensure the read completes before subsequent
    // ops use it.
    rewriter.create<ttmetal::FinishOp>(op->getLoc());
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

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateD2MToTTMetalPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter & /*typeConverter*/,
                                  ttmetal::MathFidelity mathFidelity) {
  patterns.add<ttmetal::MemrefAllocRewriter, ttmetal::MemrefDeallocRewriter,
               ttmetal::D2MToDeviceRewriter, ttmetal::D2MToHostRewriter,
               ttmetal::D2MMeshShardRewriter>(ctx);
  patterns.add<ttmetal::D2MGenericRewriter>(ctx, mathFidelity);
}

} // namespace mlir::tt
