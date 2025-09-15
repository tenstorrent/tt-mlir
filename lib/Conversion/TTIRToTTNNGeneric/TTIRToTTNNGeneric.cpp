// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNNGeneric/TTIRToTTNNGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {

namespace {
class TTIRGenericRewriter : public OpConversionPattern<ttir::GenericOp> {
public:
  using OpConversionPattern<ttir::GenericOp>::OpConversionPattern;

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
    bool isReadKernel = true;
    for (const auto [i, thread] : llvm::enumerate(threads)) {
      const ttir::ThreadAttr threadAttr = mlir::cast<ttir::ThreadAttr>(thread);

      // Get kernel args
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

      // Create KernelDescriptor
      switch (threadAttr.getThreadType()) {
      case ttir::ThreadType::Compute: {
        // TODO (vtangTT): support lowering to different compute configs.
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
      // TODO (vtangTT): fix this assumption that order is read->compute->write
      // so just flip isReadKernel flag every iteration
      case ttir::ThreadType::Datamovement: {
        if (isReadKernel) {
          kernelConfigs[i] = builder.getAttr<ttnn::ReadKernelAttr>(
              kernelSymbol, coreRangeSet, kernelRTArgs, kernelCTArgs);
        } else {
          kernelConfigs[i] = builder.getAttr<ttnn::WriteKernelAttr>(
              kernelSymbol, coreRangeSet, kernelRTArgs, kernelCTArgs);
        }
        isReadKernel = !isReadKernel;
        break;
      }
      }
    }
    return kernelConfigs;
  }

  LogicalResult
  matchAndRewrite(ttir::GenericOp op, ttir::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    MLIRContext *ctx = rewriter.getContext();
    const size_t size = op.getOperands().size();
    llvm::SmallVector<Value> ios(size);
    llvm::SmallVector<Value> cbs(size);
    llvm::SmallVector<int64_t> cbPorts(size);
    int64_t cbPort = 0;
    for (const auto [i, operand] : llvm::enumerate(op.getOperands())) {
      if (auto castOp =
              mlir::dyn_cast_if_present<ttir::TTNNToMetalLayoutCastOp>(
                  operand.getDefiningOp());
          castOp) {
        // Use the TTNN tensor operand of the cast as the io for ttnn.generic,
        // Use the memref operand for CB descriptor creation.
        ios[i] = castOp->getOperands()[0];
        cbs[i] = castOp->getOperands()[1];
      } else {
        llvm_unreachable("Expected TTNNToMetalLayoutCastOp");
      }
      cbPorts[i] = cbPort++;
    }

    ttcore::GridAttr grid = op.getGrid();
    ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
        ctx, ttnn::CoreRangeAttr::get(
                 ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                 ttnn::CoreCoordAttr::get(ctx, grid.getShape()[0] - 1,
                                          grid.getShape()[1] - 1)));

    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbPort);
    if (cbs.empty()) {
      llvm_unreachable("Expected circular buffers.");
    }

    // Create CBDescriptor
    for (auto [i, cb] : llvm::enumerate(cbs)) {
      auto cb_memref = dyn_cast<MemRefType>(cb.getType());
      auto shape = cb_memref.getShape();
      auto elementSize = cb_memref.getElementType().getIntOrFloatBitWidth() / 8;
      auto pageSize = std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int64_t>()) *
                      elementSize;

      ttcore::DataType dtype =
          ttcore::elementTypeToDataType(cb_memref.getElementType());
      ttnn::KernelCBFormatAttr cbFormat =
          ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);
      // Assumption is one CBFormat per descriptor, so total_size = page_size
      ttnn::KernelCBAttr cbDescriptor =
          ttnn::KernelCBAttr::get(ctx, pageSize, coreRangeSet, {cbFormat});
      cbDescriptors[i] = cbDescriptor;
    }

    // Create KernelDescriptors
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
class TTNNToMetalLayoutCastRewriter
    : public OpConversionPattern<ttir::TTNNToMetalLayoutCastOp> {
public:
  using OpConversionPattern<ttir::TTNNToMetalLayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TTNNToMetalLayoutCastOp op,
                  ttir::TTNNToMetalLayoutCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    // Replace all uses of the cast result with the second input (result of the
    // cast)
    if (!op->getResults().empty()) {
      Value replacement = op.getOperation()->getOperand(1);
      rewriter.replaceOp(op, replacement);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  };
};
} // namespace

} // namespace mlir::tt
namespace mlir::tt {
void populateTTIRToTTNNGenericPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<TTIRGenericRewriter, TTNNToMetalLayoutCastRewriter>(ctx);
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
