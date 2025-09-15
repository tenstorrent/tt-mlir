// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNNGeneric/TTIRToTTNNGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <array>

namespace mlir::tt {

namespace {
class TTIRGenericRewriter : public OpConversionPattern<ttir::GenericOp> {
public:
  using OpConversionPattern<ttir::GenericOp>::OpConversionPattern;

  static mlir::Attribute convertKernelArg(Builder &builder,
                                          ttkernel::ArgAttr arg) {
    switch (arg.getArgType()) {
    case ttkernel::ArgType::BufferAddress: {
      return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::CBPort: {
      return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::Semaphore: {
      return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(
          arg.getOperandIndex());
    }
    }
  }

  static SmallVector<mlir::Attribute> convertThreadsToKernelConfigs(
      Builder &builder, mlir::ValueRange operands, ArrayAttr threads,
      ttnn::CoreRangeSetAttr coreRangeSet, const SymbolTable &symbolTable) {
    SmallVector<mlir::Attribute> kernelConfigs;
    // uint32_t nocIndex = 0;
    // TODO (vtangTT): fix this assumption that order is read->compute->write so
    // flip flag every iteration
    bool isReadKernel = true;
    for (Attribute thread : threads) {
      ttir::ThreadAttr threadAttr = mlir::cast<ttir::ThreadAttr>(thread);

      // Get kernel args
      SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
      auto kernelFunc = symbolTable.lookup<mlir::func::FuncOp>(
          kernelSymbol.getRootReference());
      auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);
      auto rtArgs = kernelSpec.getRtArgs();
      auto ctArgs = kernelSpec.getCtArgs();
      llvm::SmallVector<mlir::Attribute> kernelCTArgs;
      llvm::SmallVector<mlir::Attribute> kernelRTArgs;
      for (ttkernel::ArgAttr arg : rtArgs) {
        kernelRTArgs.push_back(convertKernelArg(builder, arg));
      }
      for (ttkernel::ArgAttr arg : ctArgs) {
        kernelCTArgs.push_back(convertKernelArg(builder, arg));
      }

      // Create KernelDescriptor
      switch (threadAttr.getThreadType()) {
      case ttir::ThreadType::Compute: {
        // TODO (vtangTT): support lowering to different compute configs.
        kernelConfigs.push_back(builder.getAttr<ttnn::ComputeKernelAttr>(
            kernelSymbol, coreRangeSet,
            /*math_fidelity*/ ttnn::ComputeKernelMathFidelity::HiFi4,
            /*fp32DestAccum*/ false,
            /*dst_full_sync_en*/ false,
            /*unpack_to_dest_mode*/
            ArrayRef<ttnn::ComputeKernelUnpackToDestMode>{
                ttnn::ComputeKernelUnpackToDestMode::Default},
            /*bfp8_pack_precise*/ false,
            /*math_approx_mode*/ false, kernelRTArgs, kernelCTArgs));
        break;
      }
      case ttir::ThreadType::Datamovement: {
        if (isReadKernel) {
          kernelConfigs.push_back(builder.getAttr<ttnn::ReadKernelAttr>(
              kernelSymbol, coreRangeSet, kernelRTArgs, kernelCTArgs));
        } else {
          kernelConfigs.push_back(builder.getAttr<ttnn::WriteKernelAttr>(
              kernelSymbol, coreRangeSet, kernelRTArgs, kernelCTArgs));
        }
        isReadKernel = !isReadKernel;
        break;
      }
      }
    }
    return kernelConfigs;
  }

  // TODO (vtangTT): THIS IS NOT CORRECT FIX THIS
  static ttcore::DataType getTTDataType(Type elementType) {
    if (auto it = dyn_cast<IntegerType>(elementType)) {
      switch (it.getWidth()) {
      case 32:
        return ttcore::DataType::UInt32;
      case 16:
        return ttcore::DataType::UInt16;
      case 8:
        return ttcore::DataType::UInt8;
      default:
        return ttcore::DataType::BFloat16;
      }
    }
    if (auto ft = dyn_cast<FloatType>(elementType)) {
      switch (ft.getWidth()) {
      case 32:
        return ttcore::DataType::Float32;
      case 16:
        return ttcore::DataType::BFloat16; // prefer bf16
      default:
        return ttcore::DataType::BFloat16;
      }
    }
    return ttcore::DataType::BFloat16;
  }

  LogicalResult
  matchAndRewrite(ttir::GenericOp op, ttir::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    MLIRContext *ctx = rewriter.getContext();
    llvm::SmallVector<Value> ios;
    llvm::SmallVector<Value> cbs;
    llvm::SmallVector<int64_t> cbPorts;
    int64_t cbPort = 0;
    for (auto operand : op.getOperands()) {
      if (auto castOp =
              mlir::dyn_cast_if_present<ttir::TTNNToMetalLayoutCastOp>(
                  operand.getDefiningOp());
          castOp) {
        // Use the TTNN tensor feeding the cast as the IO for ttnn.generic,
        // Use the memref operand for CB descriptor creation.
        ios.push_back(castOp->getOperands()[0]);
        cbs.push_back(castOp->getOperands()[1]);
      } else {
        llvm::errs() << "Unsupported input type: " << operand.getType() << "\n";
        return failure();
      }
      cbPorts.push_back(cbPort++);
    }

    ttcore::GridAttr grid = op.getGrid();
    ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
        ctx, ttnn::CoreRangeAttr::get(
                 ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                 ttnn::CoreCoordAttr::get(ctx, grid.getShape()[0],
                                          grid.getShape()[1])));

    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbPort);
    if (cbs.empty()) {
      llvm::errs() << "No CB found\n";
      return failure();
    }

    // Create CBDescriptor: assume only one CBFormat per descriptor
    for (auto [i, cb] : llvm::enumerate(cbs)) {
      auto cb_memref = dyn_cast<MemRefType>(cb.getType());
      auto shape = cb_memref.getShape();
      auto elementSize = cb_memref.getElementType().getIntOrFloatBitWidth() / 8;
      auto pageSize = std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int64_t>()) *
                      elementSize;

      ttcore::DataType dtype = getTTDataType(cb_memref.getElementType());
      ttnn::KernelCBFormatAttr cbFormat =
          ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);
      ttnn::KernelCBAttr cbDescriptor =
          ttnn::KernelCBAttr::get(ctx, pageSize, coreRangeSet, {cbFormat});
      cbDescriptors[i] = cbDescriptor;
    }

    // Create KernelDescriptors
    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    llvm::SmallVector<mlir::Attribute> kernelDescriptors =
        convertThreadsToKernelConfigs(rewriter, adaptor.getOperands(),
                                      op.getThreads(), coreRangeSet,
                                      opSymTable);

    llvm::SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors;

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        ctx, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

    rewriter.replaceOpWithNewOp<ttnn::GenericOp>(op, ios, program,
                                                 ttnn::MemoryConfigAttr());
    return success();
  };
};
} // namespace

namespace {
class TTNNToMetalLayoutCastRewriter
    : public OpConversionPattern<ttir::TTNNToMetalLayoutCastOp> {
public:
  using OpConversionPattern<ttir::TTNNToMetalLayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TTNNToMetalLayoutCastOp op,
                  ttir::TTNNToMetalLayoutCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (!op->getResults().empty()) {
      // Replace all uses of the cast result with the second input (result of
      // the cast)
      Value replacement = op.getOperation()->getOperand(1);
      rewriter.replaceOp(op, replacement);
    } else {
      rewriter.eraseOp(op);
    }
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
    rewriter.eraseOp(op);
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

    rewriter.eraseOp(op);
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

    llvm::errs() << "Should not be lowered to ToLayoutOp\n";
    return success();
  }
};

} // namespace
} // namespace mlir::tt
namespace mlir::tt {
void populateTTIRToTTNNGenericPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<TTIRGenericRewriter, MemrefAllocRewriter, MemrefDeallocRewriter,
               TTIRToLayoutRewriter, TTNNToMetalLayoutCastRewriter>(ctx);
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
