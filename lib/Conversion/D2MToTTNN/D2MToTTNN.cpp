// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {

namespace {
class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  using OpConversionPattern<d2m::GenericOp>::OpConversionPattern;

  static mlir::Attribute convertKernelArg(Builder &builder,
                                          const ttkernel::ArgAttr &arg) {
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

  static SmallVector<mlir::Attribute>
  convertThreadsToKernelConfigs(Builder &builder, const ArrayAttr &threads,
                                const ttnn::CoreRangeSetAttr &coreRangeSet,
                                const SymbolTable &symbolTable) {
    SmallVector<mlir::Attribute> kernelConfigs(threads.size());
    int nocIndex = 0;
    for (const auto [i, thread] : llvm::enumerate(threads)) {
      const d2m::ThreadAttr threadAttr = mlir::cast<d2m::ThreadAttr>(thread);

      // Get kernel args.
      SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
      auto kernelFunc = symbolTable.lookup<mlir::func::FuncOp>(
          kernelSymbol.getRootReference());
      auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);
      auto rtArgs = kernelSpec.getRtArgs();
      auto ctArgs = kernelSpec.getCtArgs();
      llvm::SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
      llvm::SmallVector<mlir::Attribute> kernelRTArgs(rtArgs.size());
      for (const auto [i, arg] : llvm::enumerate(rtArgs)) {
        kernelRTArgs[i] = convertKernelArg(builder, arg);
      }
      for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
        kernelCTArgs[i] = convertKernelArg(builder, arg);
      }

      // Create KernelDescriptor.
      switch (threadAttr.getThreadType()) {
      case d2m::ThreadType::Compute: {
        // TODO (vtangTT) #5032: support lowering to different compute configs.
        kernelConfigs[i] = builder.getAttr<ttnn::ComputeKernelAttr>(
            kernelSymbol, coreRangeSet,
            /*math_fidelity*/ ttnn::ComputeKernelMathFidelity::HiFi4,
            /*fp32DestAccum*/ false,
            /*dst_full_sync_en*/ false,
            /*unpack_to_dest_mode*/
            ArrayRef<ttnn::ComputeKernelUnpackToDestMode>{
                ttnn::ComputeKernelUnpackToDestMode::Default},
            /*bfp8_pack_precise*/ false,
            /*math_approx_mode*/ false, kernelRTArgs, kernelCTArgs);
        break;
      }
      // TODO (vtangTT) #5033: fix this assumption that order is
      // read->write->compute; nocIndex == 0 for read, nocIndex == 1 for write.
      case d2m::ThreadType::Datamovement: {
        TT_assert(nocIndex < 2);
        if (nocIndex == 0) {
          kernelConfigs[i] = builder.getAttr<ttnn::ReadKernelAttr>(
              kernelSymbol, coreRangeSet, kernelRTArgs, kernelCTArgs);
        } else {
          kernelConfigs[i] = builder.getAttr<ttnn::WriteKernelAttr>(
              kernelSymbol, coreRangeSet, kernelRTArgs, kernelCTArgs);
        }
        nocIndex++;
        break;
      }
      }
    }
    return kernelConfigs;
  }

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    MLIRContext *ctx = rewriter.getContext();
    const size_t size = op.getOperands().size();
    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    ttcore::GridAttr grid = op.getGrid();
    ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
        ctx, ttnn::CoreRangeAttr::get(
                 ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                 ttnn::CoreCoordAttr::get(ctx, grid.getShape()[0] - 1,
                                          grid.getShape()[1] - 1)));

    llvm::SmallVector<Value> ios(size);
    llvm::SmallVector<Value> cbs(size);
    llvm::SmallVector<int64_t> cbPorts(size);
    int64_t cbPort = 0;
    const size_t numInputs = op.getInputs().size();
    for (auto [i, operand] : llvm::enumerate(op.getInputs())) {
      if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
              operand.getDefiningOp());
          streamLayoutOp) {
        if (auto castOp =
                mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
                    streamLayoutOp.getInput().getDefiningOp());
            castOp) {
          ios[i] = castOp.getOperand();
        } else {
          llvm_unreachable(
              "Expected TTNNMetalLayoutCastOp producing stream input.");
        }
        cbs[i] = streamLayoutOp.getStorage();
      } else {
        llvm_unreachable("Expected stream_layout op for the input.");
      }
      cbPorts[i] = cbPort++;
    }
    for (const auto [i, operand] : llvm::enumerate(op.getOutputs())) {
      const size_t idx = numInputs + i;
      if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
              operand.getDefiningOp());
          castOp) {
        // Use the TTNN tensor operand of the cast as the output for
        // ttnn.generic, Use the memref result for CB descriptor creation.
        ios[idx] = castOp->getOperands()[0];
        cbs[idx] = castOp->getResult(0);
      } else {
        llvm_unreachable("Expected TTNNToMetalLayoutCastOp");
      }
      cbPorts[idx] = cbPort++;
    }

    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbPort);
    if (cbs.empty()) {
      llvm_unreachable("Expected circular buffers.");
    }

    // Create CBDescriptor.
    // TODO (vtangTT) #5031: support setting buffer ptr in CBDescriptor for
    // aliasing.
    for (auto [i, cb] : llvm::enumerate(cbs)) {
      auto cb_memref = dyn_cast<MemRefType>(cb.getType());

      TT_assertv(mlir::isa<ttcore::TileType>(cb_memref.getElementType()),
                 "Only TileType supported.");
      ttcore::DataType dtype =
          ttcore::elementTypeToDataType(cb_memref.getElementType());
      size_t pageSize = device.getMemrefCBPageSizeBytes(cb_memref);
      size_t numPages = device.getMemrefCBNumPages(cb_memref);

      ttnn::KernelCBFormatAttr cbFormat =
          ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);
      ttnn::KernelCBAttr cbDescriptor = ttnn::KernelCBAttr::get(
          ctx, numPages * pageSize, coreRangeSet, {cbFormat});
      cbDescriptors[i] = cbDescriptor;
    }

    // Create KernelDescriptors.
    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    llvm::SmallVector<mlir::Attribute> kernelDescriptors =
        convertThreadsToKernelConfigs(rewriter, op.getThreads(), coreRangeSet,
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
class TTNNMetalLayoutCastRewriter
    : public OpConversionPattern<ttir::TTNNMetalLayoutCastOp> {
public:
  using OpConversionPattern<ttir::TTNNMetalLayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TTNNMetalLayoutCastOp op,
                  ttir::TTNNMetalLayoutCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto inner =
            op.getOperand().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      rewriter.replaceOp(op, inner.getOperand());
    }
    return success();
  };
};
} // namespace

namespace {
class StreamLayoutRewriter : public OpConversionPattern<d2m::StreamLayoutOp> {
public:
  using OpConversionPattern<d2m::StreamLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::StreamLayoutOp op, d2m::StreamLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (auto castOp =
            op.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      rewriter.replaceAllUsesWith(op, castOp.getOperand());
      rewriter.eraseOp(castOp);
    } else {
      llvm_unreachable("Expected TTNNMetalLayoutCastOp as stream input.");
    }

    // Canonicalization will clean up dead inputs of stream_layout.
    rewriter.eraseOp(op);
    return success();
  };
};
} // namespace

namespace {
class D2MEmptyRewriter : public OpConversionPattern<d2m::EmptyOp> {
public:
  using OpConversionPattern<d2m::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::EmptyOp op, d2m::EmptyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto layoutAttr = cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());
    auto shape = ttnn::ShapeAttr::get(ctx, tensorType.getShape());
    auto dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
    auto layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);
    auto memcfg =
        ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(op, tensorType, device, shape,
                                               dtype, layout, memcfg);
    return success();
  };
};
} // namespace

} // namespace mlir::tt
namespace mlir::tt {
void populateD2MToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter) {
  patterns.add<D2MGenericRewriter, TTNNMetalLayoutCastRewriter,
               D2MEmptyRewriter, StreamLayoutRewriter>(ctx);
}

} // namespace mlir::tt
