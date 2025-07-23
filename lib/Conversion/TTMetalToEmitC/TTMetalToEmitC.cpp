// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTMetalToEmitC/TTMetalToEmitC.h"

#include "ttmlir/Conversion/TTMetalToEmitC/EmitCConversion.h"
#include "ttmlir/Conversion/TTMetalToEmitC/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace mlir;
using namespace mlir::tt;

// Base class for TTMetal to EmitC OpConversionPattern
namespace {
template <typename SourceOp>
class TTMetalToEmitCBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using Adaptor = typename SourceOp::Adaptor;

private:
  std::string virtual getPrefixSearchPattern() const { return "ttmetal."; }
  std::string virtual getPrefixSwapPattern() const { return "tt::tt_metal::"; }

public:
  // Converts op name by removing the dialect prefix ("ttmetal.") and replacing
  // with namespace prefix ("tt::tt_metal::")
  std::string convertOpName(SourceOp op) const {
    auto name = op.getOperationName();
    assert(
        name.starts_with(getPrefixSearchPattern()) &&
        "TTMetalToEmitCBaseOpConversionPattern only supports ops from the TTMetal dialect");

    return name.str().replace(0, getPrefixSearchPattern().size(),
                              getPrefixSwapPattern());
  }
};
} // namespace

// EnqueueProgram operation conversion
namespace {
class EnqueueProgramOpConversionPattern
    : public TTMetalToEmitCBaseOpConversionPattern<ttmetal::EnqueueProgramOp> {

public:
  using TTMetalToEmitCBaseOpConversionPattern<
      ttmetal::EnqueueProgramOp>::TTMetalToEmitCBaseOpConversionPattern;
  using Adaptor = ttmetal::EnqueueProgramOp::Adaptor;

  LogicalResult
  matchAndRewrite(ttmetal::EnqueueProgramOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttmetal_to_emitc::EmitCTTMetalEmitter<ttmetal::EnqueueProgramOp> emitter(
        srcOp, adaptor, rewriter);

    // Get the kernel configurations
    auto kernelConfigs = srcOp.getKernelConfigs();
    
    llvm::SmallVector<mlir::Attribute> args;
    
    // Add buffer operands
    for (auto buffer : srcOp.getBuffers()) {
      args.push_back(emitter.emit(buffer));
    }
    
    // Add circular buffer operands
    for (auto cb : srcOp.getCbs()) {
      args.push_back(emitter.emit(cb));
    }
    
    // Add CB ports (as array)
    args.push_back(emitter.emit<std::vector<int64_t>>(srcOp.getCbPortsAttr()));
    
    // Add kernel configurations (as array)
    args.push_back(emitter.emit<std::vector<::tt::tt_metal::ComputeConfig>>(kernelConfigs));

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EnqueueWriteBuffer operation conversion
namespace {
class EnqueueWriteBufferOpConversionPattern
    : public TTMetalToEmitCBaseOpConversionPattern<ttmetal::EnqueueWriteBufferOp> {

public:
  using TTMetalToEmitCBaseOpConversionPattern<
      ttmetal::EnqueueWriteBufferOp>::TTMetalToEmitCBaseOpConversionPattern;
  using Adaptor = ttmetal::EnqueueWriteBufferOp::Adaptor;

  LogicalResult
  matchAndRewrite(ttmetal::EnqueueWriteBufferOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttmetal_to_emitc::EmitCTTMetalEmitter<ttmetal::EnqueueWriteBufferOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getOutput()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// EnqueueReadBuffer operation conversion
namespace {
class EnqueueReadBufferOpConversionPattern
    : public TTMetalToEmitCBaseOpConversionPattern<ttmetal::EnqueueReadBufferOp> {

public:
  using TTMetalToEmitCBaseOpConversionPattern<
      ttmetal::EnqueueReadBufferOp>::TTMetalToEmitCBaseOpConversionPattern;
  using Adaptor = ttmetal::EnqueueReadBufferOp::Adaptor;

  LogicalResult
  matchAndRewrite(ttmetal::EnqueueReadBufferOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ttmetal_to_emitc::EmitCTTMetalEmitter<ttmetal::EnqueueReadBufferOp> emitter(
        srcOp, adaptor, rewriter);

    llvm::SmallVector<mlir::Attribute> args{
        emitter.emit(srcOp.getInput()),
        emitter.emit(srcOp.getOutput()),
    };

    emitter.replaceOp(*this, args);

    return success();
  }
};
} // namespace

// MemRef allocation to Buffer creation conversion
namespace {
class AllocOpConversionPattern
    : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    // Insert runtime helper functions if not exists
    ttmetal_to_emitc::utils::insertRuntimeHelperFunctionsIfNotExists(rewriter, srcOp);
    
    // Restore insertion point to the current operation
    rewriter.setInsertionPoint(srcOp);
    
    // Get the memref type information
    auto memrefType = srcOp.getType();
    auto shape = memrefType.getShape();
    auto elementType = memrefType.getElementType();
    
    // Calculate total size in bytes
    int64_t totalElements = 1;
    for (auto dim : shape) {
      if (dim != ShapedType::kDynamic) {
        totalElements *= dim;
      }
    }
    
    // Get element size in bytes (simplified)
    int64_t elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;
    int64_t totalSizeBytes = totalElements * elementSizeBytes;
    
    // Create buffer allocation call
    auto bufferType = emitc::PointerType::get(
        emitc::OpaqueType::get(rewriter.getContext(), 
                               ttmetal_to_emitc::TypeNameV<::tt::tt_metal::Buffer>));
    
    auto sizeAttr = rewriter.getI64IntegerAttr(totalSizeBytes);
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, bufferType,
        ttmetal_to_emitc::utils::kCreateBufferFunctionName,
        rewriter.getArrayAttr({sizeAttr}),
        /*template_args=*/nullptr,
        ValueRange{});

    return success();
  }
};
} // namespace

// MemRef deallocation to Buffer cleanup conversion  
namespace {
class DeallocOpConversionPattern
    : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    // Create buffer deallocation call
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, TypeRange{},
        ttmetal_to_emitc::utils::kDeallocateBufferFunctionName,
        /*args=*/nullptr,
        /*template_args=*/nullptr,
        adaptor.getOperands());

    return success();
  }
};
} // namespace

// Populate all conversion patterns
void mlir::tt::populateTTMetalToEmitCPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  // TTMetal operation patterns
  patterns.add<EnqueueProgramOpConversionPattern>(typeConverter, ctx);
  patterns.add<EnqueueWriteBufferOpConversionPattern>(typeConverter, ctx);
  patterns.add<EnqueueReadBufferOpConversionPattern>(typeConverter, ctx);
  
  // MemRef operation patterns for buffer management
  patterns.add<AllocOpConversionPattern>(typeConverter, ctx);
  patterns.add<DeallocOpConversionPattern>(typeConverter, ctx);
}