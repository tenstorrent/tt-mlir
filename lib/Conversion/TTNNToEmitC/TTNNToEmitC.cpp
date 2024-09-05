// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace {

// Base class for TTNN to EmitC OpConversionPattern
//
template <typename SourceOp>
class TTNNToEmitCBaseOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  TTNNToEmitCBaseOpConversionPattern(const TypeConverter &typeConverter,
                                     MLIRContext *context,
                                     PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit) {}

private:
  std::string virtual getPrefixSearchPattern() const { return "ttnn."; }
  std::string virtual getPrefixSwapPattern() const { return "ttnn::"; }

public:
  // Converts op name by removing the dialect prefix ("ttnn.") and replacing
  // with namespace prefix ("ttnn::")
  //
  std::string convertOpName(SourceOp op) const {
    auto name = op.getOperationName();
    assert(
        name.starts_with(getPrefixSearchPattern()) &&
        "DefaultOpConversionPattern only supports ops from the TTNN dialect");

    return name.str().replace(0, getPrefixSearchPattern().size(),
                              getPrefixSwapPattern());
  }
};

// Default op conversion pattern, used to convert most ops
//
template <typename SourceOp, typename Adaptor = typename SourceOp::Adaptor>
class DefaultOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<SourceOp> {

public:
  DefaultOpConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<SourceOp>(typeConverter, context,
                                                     benefit) {}

  LogicalResult
  matchAndRewrite(SourceOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int numReturnTypes = srcOp->getResultTypes().size();
    assert(numReturnTypes <= 1 &&
           "DefaultOpConversionPattern does not support multiple return types");

    // If srcOp has a return type, cast it before converting
    //
    if (numReturnTypes == 1) {
      auto resultTy = cast<emitc::OpaqueType>(
          this->getTypeConverter()->convertType(srcOp->getResult(0).getType()));
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, resultTy, this->convertOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    } else {
      // No return type, only convert the op
      //
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp), nullptr,
          nullptr, adaptor.getOperands());
    }

    return success();
  }
};

// OpenDeviceOp conversion pattern
//
class OpenDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::OpenDeviceOp> {

public:
  OpenDeviceOpConversionPattern(const TypeConverter &typeConverter,
                                MLIRContext *context,
                                PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::OpenDeviceOp>(
            typeConverter, context, benefit) {}

public:
  LogicalResult
  matchAndRewrite(ttnn::OpenDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy =
        this->getTypeConverter()->convertType(srcOp->getResult(0).getType());

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, resultTy, this->convertOpName(srcOp), srcOp.getDeviceIdsAttr(),
        nullptr, adaptor.getOperands());

    return success();
  }
};

// CloseDeviceOp conversion pattern
//
class CloseDeviceOpConversionPattern
    : public TTNNToEmitCBaseOpConversionPattern<ttnn::CloseDeviceOp> {

public:
  CloseDeviceOpConversionPattern(const TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : TTNNToEmitCBaseOpConversionPattern<ttnn::CloseDeviceOp>(
            typeConverter, context, benefit) {}

public:
  LogicalResult
  matchAndRewrite(ttnn::CloseDeviceOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        srcOp, srcOp->getResultTypes(), this->convertOpName(srcOp), nullptr,
        nullptr, adaptor.getOperands());

    return success();
  }
};

} // namespace

namespace mlir::tt {

void populateTTNNToEmitCPatterns(mlir::MLIRContext *ctx,
                                 mlir::RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // Device ops
  //
  patterns.add<OpenDeviceOpConversionPattern>(typeConverter, ctx);
  patterns.add<CloseDeviceOpConversionPattern>(typeConverter, ctx);

  // Memory ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::ToMemoryConfigOp>>(
      typeConverter, ctx);

  // Tensor ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::EmptyOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::FullOp>>(typeConverter, ctx);

  // Eltwise unary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::ReluOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SqrtOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SigmoidOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ReciprocalOp>>(typeConverter,
                                                               ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ExpOp>>(typeConverter, ctx);

  // Eltwise binary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AddOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SubtractOp>>(typeConverter,
                                                             ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MultiplyOp>>(typeConverter,
                                                             ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::GreaterEqualOp>>(typeConverter,
                                                                 ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::DivOp>>(typeConverter, ctx);

  // Tensor manipulation ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::TransposeOp>>(typeConverter,
                                                              ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ConcatOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::ReshapeOp>>(typeConverter, ctx);

  // Matmul ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::MatmulOp>>(typeConverter, ctx);

  // Reduction ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::SumOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MeanOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MaxOp>>(typeConverter, ctx);

  // Conv ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::Conv2dOp>>(typeConverter, ctx);

  // Other ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::SoftmaxOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::EmbeddingOp>>(typeConverter,
                                                              ctx);
}

} // namespace mlir::tt
