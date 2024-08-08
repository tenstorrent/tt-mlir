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

// Default op conversion pattern, used to convert most ops
//
template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class DefaultOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  DefaultOpConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<SrcOp>(typeConverter, context, benefit) {}

  // Converts op name by removing the dialect prefix ("ttnn.") and replacing
  // with namespace prefix ("ttnn::")
  //
  std::string convertOpName(SrcOp op) const {
    auto name = op.getOperationName();
    assert(
        name.starts_with("ttnn.") &&
        "DefaultOpConversionPattern only supports ops from the TTNN dialect");

    return name.str().replace(0, 5, "ttnn::");
  }

  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
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
          srcOp, resultTy, convertOpName(srcOp), nullptr, nullptr,
          adaptor.getOperands());
    } else {
      // No return type, only convert the op
      //
      rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          srcOp, srcOp->getResultTypes(), convertOpName(srcOp), nullptr,
          nullptr, adaptor.getOperands());
    }

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
  patterns.add<DefaultOpConversionPattern<ttnn::OpenDeviceOp>>(typeConverter,
                                                               ctx, true);
  patterns.add<DefaultOpConversionPattern<ttnn::CloseDeviceOp>>(typeConverter,
                                                                ctx);

  // Memory ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::ToMemoryConfigOp>>(
      typeConverter, ctx);

  // Tensor ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::FullOp>>(typeConverter, ctx);

  // Eltwise unary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::ReluOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SoftmaxOp>>(typeConverter, ctx);

  // Eltwise binary ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::AddOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::SubtractOp>>(typeConverter,
                                                             ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MultiplyOp>>(typeConverter,
                                                             ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::GreaterEqualOp>>(typeConverter,
                                                                 ctx);

  // Tensor manipulation ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::TransposeOp>>(typeConverter,
                                                              ctx);

  // Matmul ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::MatmulOp>>(typeConverter, ctx);

  // Reduction ops
  //
  patterns.add<DefaultOpConversionPattern<ttnn::SumOp>>(typeConverter, ctx);
  patterns.add<DefaultOpConversionPattern<ttnn::MeanOp>>(typeConverter, ctx);
}

} // namespace mlir::tt
