// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Utils/Mesh.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {
template <typename SrcOp, typename DestOp>
class TTNNToTTIRElementwiseConversionPattern
    : public mlir::OpConversionPattern<SrcOp> {

  using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;

public:
  mlir::LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (mlir::tt::ttnn::utils::isTTNNHoistGenericViaD2MOp(srcOp)) {
      auto outputType = mlir::cast<mlir::RankedTensorType>(
          this->getTypeConverter()->convertType(srcOp.getResult().getType()));

      mlir::tt::ttir::utils::replaceOpWithNewDPSOp<DestOp>(
          rewriter, srcOp, outputType, adaptor.getOperands());
    }
    return mlir::success();
  }
};
} // namespace

static void
addElementwiseUnaryOpsConversionPatterns(mlir::MLIRContext *ctx,
                                         mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &typeConverter) {

  patterns
      .add<TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::AbsOp,
                                                  mlir::tt::ttir::AbsOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::CbrtOp,
                                                  mlir::tt::ttir::CbrtOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::TypecastOp,
                                                  mlir::tt::ttir::TypecastOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::CeilOp,
                                                  mlir::tt::ttir::CeilOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::CosOp,
                                                  mlir::tt::ttir::CosOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::ExpOp,
                                                  mlir::tt::ttir::ExpOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::FloorOp,
                                                  mlir::tt::ttir::FloorOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::IsFiniteOp,
                                                  mlir::tt::ttir::IsFiniteOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::NegOp,
                                                  mlir::tt::ttir::NegOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::RsqrtOp,
                                                  mlir::tt::ttir::RsqrtOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SinOp,
                                                  mlir::tt::ttir::SinOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SqrtOp,
                                                  mlir::tt::ttir::SqrtOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::Log1pOp,
                                                  mlir::tt::ttir::Log1pOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::Expm1Op,
                                                  mlir::tt::ttir::Expm1Op>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SignOp,
                                                  mlir::tt::ttir::SignOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SigmoidOp,
                                                  mlir::tt::ttir::SigmoidOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::TanOp,
                                                  mlir::tt::ttir::TanOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::TanhOp,
                                                  mlir::tt::ttir::TanhOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::LogOp,
                                                  mlir::tt::ttir::LogOp>>(
          typeConverter, ctx);
}

static void
addElementwiseBinaryOpsConversionPatterns(mlir::MLIRContext *ctx,
                                          mlir::RewritePatternSet &patterns,
                                          mlir::TypeConverter &typeConverter) {

  patterns
      .add<TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::AddOp,
                                                  mlir::tt::ttir::AddOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::DivideOp,
                                                  mlir::tt::ttir::DivOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::MaxOp,
                                                  mlir::tt::ttir::MaximumOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::MinOp,
                                                  mlir::tt::ttir::MinimumOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::MultiplyOp,
                                                  mlir::tt::ttir::MultiplyOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::SubtractOp,
                                                  mlir::tt::ttir::SubtractOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::RemainderOp,
                                                  mlir::tt::ttir::RemainderOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::WhereOp,
                                                  mlir::tt::ttir::WhereOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::PowOp,
                                                  mlir::tt::ttir::PowOp>,
           TTNNToTTIRElementwiseConversionPattern<mlir::tt::ttnn::Atan2Op,
                                                  mlir::tt::ttir::Atan2Op>,
           TTNNToTTIRElementwiseConversionPattern<
               mlir::tt::ttnn::LogicalRightShiftOp,
               mlir::tt::ttir::LogicalRightShiftOp>,
           TTNNToTTIRElementwiseConversionPattern<
               mlir::tt::ttnn::LogicalLeftShiftOp,
               mlir::tt::ttir::LogicalLeftShiftOp>>(typeConverter, ctx);
}

namespace mlir::tt {

void populateTTNNToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
