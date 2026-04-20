// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/Builders.h"
#include <cstdint>
#include <mlir-c/IR.h>

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

  static KernelArgsAttr
  evalKernelArgsFromSpec(Builder &builder, const SymbolTable &symbolTable,
                         SymbolRefAttr kernelSymbol,
                         const DenseMap<size_t, size_t> &cbOperandIndexToPort) {
    auto kernelFunc =
        symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
    ttkernel::ArgSpecAttr kernelSpec =
        kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
            ttkernel::ArgSpecAttr::name);
    SmallVector<ttmetal::KernelArgAttr> rtArgs;
    SmallVector<ttmetal::KernelArgAttr> ctArgs;
    for (ttkernel::ArgAttr arg : kernelSpec.getRtArgs()) {
      if (arg.getArgType() == ttkernel::ArgType::CB) {
        rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), cbOperandIndexToPort.at(arg.getOperandIndex())));
      } else {
        rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), arg.getOperandIndex()));
      }
    }
    for (ttkernel::ArgAttr arg : kernelSpec.getCtArgs()) {
      if (arg.getArgType() == ttkernel::ArgType::CB) {
        ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), cbOperandIndexToPort.at(arg.getOperandIndex())));
      } else {
        ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), arg.getOperandIndex()));
      }
    }
    return builder.getAttr<ttmetal::KernelArgsAttr>(rtArgs, ctArgs);
  }

  static ArrayAttr convertThreadsToKernelConfigs(
      Builder &builder, mlir::ValueRange inputOutputOperands, ArrayAttr threads,
      ArrayRef<int64_t> physicalGridShape, const SymbolTable &symbolTable,
      ttmetal::MathFidelity mathFidelity,
      const DenseMap<size_t, size_t> &cbOperandIndexToPort) {
    SmallVector<Attribute> kernelConfigs;
    int unassignedNocCounter = 0;

    auto coreRange = ttmetal::CoreRangeAttr::getPhysicalCoreRange(
        builder.getContext(), physicalGridShape);

    for (Attribute threadAttr : threads) {
      d2m::ThreadAttr thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
      KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
          builder, symbolTable, thread.getKernelSymbol(), cbOperandIndexToPort);
      Attribute kernelConfig = nullptr;
      switch (thread.getThreadType()) {
      case d2m::ThreadType::Compute: {
        bool fp32DestAccum = false;
        for (size_t i = 0; i < inputOutputOperands.size(); ++i) {
          ttcore::DataType dataType = ttcore::elementTypeToDataType(
              ttcore::getOperandInnerElementType(inputOutputOperands[i]));

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
        int32_t nocIdx = thread.getNocIndex();
        if (nocIdx < 0) {
          nocIdx = unassignedNocCounter++ % 2;
        }
        kernelConfig = builder.getAttr<ttmetal::NocConfigAttr>(
            thread.getKernelSymbol(), coreRange, kernelArgs,
            *ttcore::symbolizeNocIndex(nocIdx));
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
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());

    llvm::SmallVector<Value> remappedBuffers;
    llvm::SmallVector<Value> args;
    for (unsigned i = 0; i < op.getInputsAndOutputs().size(); ++i) {
      auto operand = adaptor.getOperands()[i];

      if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
              operand.getDefiningOp());
          view) {
        args.push_back(view.getInput());
        remappedBuffers.push_back(rewriter.getRemappedValue(view.getInput()));
      } else {
        args.push_back(operand);
        remappedBuffers.push_back(rewriter.getRemappedValue(operand));
      }
    }

    // Add additional args.
    llvm::SmallVector<Value> cbs;
    llvm::SmallVector<int64_t> cbPorts;
    int64_t cbPort = 0;
    DenseMap<size_t, size_t> cbOperandIndexToPort;
    unsigned ioSize = op.getInputsAndOutputs().size();
    for (unsigned i = 0; i < op.getAdditionalArgs().size(); ++i) {
      auto operandIndex = ioSize + i;
      auto adaptedOperand = adaptor.getOperands()[operandIndex];
      if (mlir::isa<ttmetal::GlobalSemaphoreType>(adaptedOperand.getType())) {
        args.push_back(adaptedOperand);
      } else if (auto memrefType = mlir::dyn_cast_if_present<MemRefType>(
                     adaptedOperand.getType());
                 memrefType) {
        assert(mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout()) &&
               "expected cb layout");
        // Hoisted CB buffer (already converted to CreateBufferOp by
        // MemrefAllocRewriter).
        if (auto aliasOp = mlir::dyn_cast<d2m::OperandAliasOp>(
                op.getOperands()[operandIndex].getDefiningOp())) {
          // OperandAliasOp's input is the generic's operand that this CB
          // aliases. It could be a function argument of the parent func or an
          // AllocOp.
          Value aliasedMemref = aliasOp.getMemref();
          auto parentFunc = op->getParentOfType<func::FuncOp>();
          bool isFuncArg =
              mlir::isa<BlockArgument>(aliasedMemref) &&
              mlir::cast<BlockArgument>(aliasedMemref).getOwner() ==
                  &parentFunc.getBody().front();
          assert((isFuncArg ||
                  mlir::isa<memref::AllocOp>(aliasedMemref.getDefiningOp())) &&
                 "expected OperandAliasOp input to be a func argument or "
                 "memref::AllocOp");
          cbs.push_back(adaptedOperand);
          cbOperandIndexToPort[operandIndex] = cbPort;
          cbPorts.push_back(cbPort++);
        } else if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
                       op.getOperands()[operandIndex].getDefiningOp());
                   allocOp) {
          cbs.push_back(adaptedOperand);
          cbOperandIndexToPort[operandIndex] = cbPort;
          cbPorts.push_back(cbPort++);
        } else {
          llvm_unreachable("expected alloc or aliad op for cb memref");
        }
      } else {
        op.emitOpError(
            "unexpected operand type in d2m.generic's additionalArgs: ")
            << adaptedOperand.getType();
        return failure();
      }
    }

    ArrayAttr threads = op.getThreads();
    auto physicalGridShape = op.getPhysicalGridShape();
    auto kernelConfigs = convertThreadsToKernelConfigs(
        rewriter, op.getInputsAndOutputs(), threads, physicalGridShape,
        symbolTable, mathFidelity_, cbOperandIndexToPort);
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, args, cbs, cbPorts, kernelConfigs,
        op.getFabricConnectionConfigAttr());
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
    if (!address) {
      return failure();
    }

    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memory space found, failing.");
    auto memrefType = op.getMemref().getType();

    auto vgm = op->getAttrOfType<AffineMapAttr>(
        d2m::utils::kVirtualGridInverseMappingAttr);
    auto fwd = op->getAttrOfType<AffineMapAttr>(
        d2m::utils::kVirtualGridForwardMappingAttr);

    // Hoisted CB allocs carry CBLayoutAttr (per-core local shape).
    // Keep the original type on CreateBufferOp so the dialect conversion
    // framework doesn't see a type mismatch.
    if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
      auto cbForOperandAttr =
          op->getAttrOfType<IntegerAttr>("d2m.cb_for_operand");
      auto cbOp = rewriter.replaceOpWithNewOp<ttmetal::CreateBufferOp>(
          op, memrefType, address, /*virtualGridInverseMapping=*/vgm,
          /*virtualGridForwardMapping=*/fwd);
      if (cbForOperandAttr) {
        cbOp->setAttr("d2m.cb_for_operand", cbForOperandAttr);
      }
      return success();
    }

    assert((mlir::isa<ttcore::ShardLayoutAttr, ttcore::InterleavedLayoutAttr>(
               memrefType.getLayout())) &&
           "expected physical device layout (shard or interleaved)");

    rewriter.replaceOpWithNewOp<ttmetal::CreateBufferOp>(
        op, memrefType, address, /*virtualGridInverseMapping=*/vgm,
        /*virtualGridForwardMapping=*/fwd);

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

namespace {
class D2MCreateGlobalSemaphoreRewriter
    : public OpConversionPattern<d2m::CreateGlobalSemaphoreOp> {
public:
  using OpConversionPattern<d2m::CreateGlobalSemaphoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::CreateGlobalSemaphoreOp op,
                  d2m::CreateGlobalSemaphoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto allocOp = op.getInput().getDefiningOp<memref::AllocOp>();
    assert(
        allocOp &&
        "No memref alloc found for CreateGlobalSemaphoreOp's input, failing.");

    // Get address from memref consumed and pass it to create global semaphore
    // op.
    auto address = allocOp->getAttrOfType<IntegerAttr>("address");
    // Get core range from memref shape.
    auto coreRange = ttmetal::CoreRangeAttr::getPhysicalCoreRange(
        rewriter.getContext(), ttcore::getGridShape(op.getInput()));
    rewriter.replaceOpWithNewOp<ttmetal::CreateGlobalSemaphoreOp>(
        op, ttmetal::GlobalSemaphoreType::get(rewriter.getContext()), address,
        adaptor.getValueAttr(), coreRange);
    return success();
  }
};
} // namespace

namespace {
class D2MResetGlobalSemaphoreRewriter
    : public OpConversionPattern<d2m::ResetGlobalSemaphoreOp> {
public:
  using OpConversionPattern<d2m::ResetGlobalSemaphoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ResetGlobalSemaphoreOp op,
                  d2m::ResetGlobalSemaphoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto createGlobalSemaphoreOp =
        op.getSemaphore().getDefiningOp<d2m::CreateGlobalSemaphoreOp>();
    assert(createGlobalSemaphoreOp &&
           "No create global semaphore op found for ResetGlobalSemaphoreOp's "
           "input, failing.");

    rewriter.replaceOpWithNewOp<ttmetal::ResetGlobalSemaphoreOp>(
        op, adaptor.getSemaphore(), adaptor.getValueAttr());
    return success();
  }
};
} // namespace

namespace {
class D2MViewLayoutRewriter : public OpConversionPattern<d2m::ViewLayoutOp> {
public:
  using OpConversionPattern<d2m::ViewLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ViewLayoutOp op, d2m::ViewLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Erase views.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};
} // namespace

namespace {
class D2MOperandAliasRewriter
    : public OpConversionPattern<d2m::OperandAliasOp> {
public:
  using OpConversionPattern<d2m::OperandAliasOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::OperandAliasOp op, d2m::OperandAliasOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::OperandAliasOp>(
        op, op.getResult().getType(), adaptor.getMemref());
    return success();
  }
};
} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateD2MToTTMetalPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter & /*typeConverter*/,
                                  ttmetal::MathFidelity mathFidelity) {
  patterns.add<
      ttmetal::MemrefAllocRewriter, ttmetal::MemrefDeallocRewriter,
      ttmetal::D2MToDeviceRewriter, ttmetal::D2MToHostRewriter,
      ttmetal::D2MMeshShardRewriter, ttmetal::D2MCreateGlobalSemaphoreRewriter,
      ttmetal::D2MResetGlobalSemaphoreRewriter, ttmetal::D2MViewLayoutRewriter>(
      ctx);
  patterns.add<ttmetal::D2MGenericRewriter>(ctx, mathFidelity);

  // remove alias op after generic conversion
  patterns.add<ttmetal::D2MOperandAliasRewriter>(ctx);
}

} // namespace mlir::tt
