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

using namespace mlir;
using namespace mlir::tt;

namespace {
template <typename SrcOp, typename DestOp,
          typename Adaptor = typename SrcOp::Adaptor>
class TTNNToTTIROpDefaultConversionPattern : public OpConversionPattern<SrcOp> {

  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ttnn::utils::isTTNNHoistGenericViaD2MOp(srcOp)) {
      auto outputType = mlir::cast<RankedTensorType>(
          this->getTypeConverter()->convertType(srcOp.getResult().getType()));

      ttir::utils::replaceOpWithNewDPSOp<DestOp>(rewriter, srcOp, outputType,
                                                 adaptor.getOperands());
    }
    return success();
  }
};
} // namespace

static void
addElementwiseUnaryOpsConversionPatterns(MLIRContext *ctx,
                                         RewritePatternSet &patterns,
                                         TypeConverter &typeConverter) {

  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::AbsOp,
                                                    mlir::tt::ttir::AbsOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::CbrtOp,
                                                    mlir::tt::ttir::CbrtOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::TypecastOp, mlir::tt::ttir::TypecastOp>>(typeConverter,
                                                               ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::CeilOp,
                                                    mlir::tt::ttir::CeilOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::CosOp,
                                                    mlir::tt::ttir::CosOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::ExpOp,
                                                    mlir::tt::ttir::ExpOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::FloorOp,
                                                    mlir::tt::ttir::FloorOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::IsFiniteOp, mlir::tt::ttir::IsFiniteOp>>(typeConverter,
                                                               ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::NegOp,
                                                    mlir::tt::ttir::NegOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::RsqrtOp,
                                                    mlir::tt::ttir::RsqrtOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::SinOp,
                                                    mlir::tt::ttir::SinOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::SqrtOp,
                                                    mlir::tt::ttir::SqrtOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::Log1pOp,
                                                    mlir::tt::ttir::Log1pOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::Expm1Op,
                                                    mlir::tt::ttir::Expm1Op>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::SignOp,
                                                    mlir::tt::ttir::SignOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::SigmoidOp,
                                                    mlir::tt::ttir::SigmoidOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::TanOp,
                                                    mlir::tt::ttir::TanOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::TanhOp,
                                                    mlir::tt::ttir::TanhOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::LogOp,
                                                    mlir::tt::ttir::LogOp>>(
      typeConverter, ctx);
}

static void
addElementwiseBinaryOpsConversionPatterns(MLIRContext *ctx,
                                          RewritePatternSet &patterns,
                                          TypeConverter &typeConverter) {

  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::AddOp,
                                                    mlir::tt::ttir::AddOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::DivideOp,
                                                    mlir::tt::ttir::DivOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::MaxOp,
                                                    mlir::tt::ttir::MaximumOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::MinOp,
                                                    mlir::tt::ttir::MinimumOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::MultiplyOp, mlir::tt::ttir::MultiplyOp>>(typeConverter,
                                                               ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::SubtractOp, mlir::tt::ttir::SubtractOp>>(typeConverter,
                                                               ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::RemainderOp, mlir::tt::ttir::RemainderOp>>(typeConverter,
                                                                 ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::WhereOp,
                                                    mlir::tt::ttir::WhereOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::PowOp,
                                                    mlir::tt::ttir::PowOp>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<mlir::tt::ttnn::Atan2Op,
                                                    mlir::tt::ttir::Atan2Op>>(
      typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::LogicalRightShiftOp,
      mlir::tt::ttir::LogicalRightShiftOp>>(typeConverter, ctx);
  patterns.add<TTNNToTTIROpDefaultConversionPattern<
      mlir::tt::ttnn::LogicalLeftShiftOp, mlir::tt::ttir::LogicalLeftShiftOp>>(
      typeConverter, ctx);
}

namespace mlir::tt {

void populateTTNNToTTIRPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  addElementwiseUnaryOpsConversionPatterns(ctx, patterns, typeConverter);
  addElementwiseBinaryOpsConversionPatterns(ctx, patterns, typeConverter);
}

} // namespace mlir::tt
