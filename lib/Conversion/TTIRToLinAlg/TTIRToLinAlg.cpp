// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinAlg/TTIRToLinAlg.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::tt;

namespace {
template <typename TTIROpTy, typename LinAlgOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes))) {
      return failure();
    }

    auto newOp = rewriter.replaceOpWithNewOp<LinAlgOpTy>(
        op, resultTypes, adaptor.getInputs(), adaptor.getOutputs());
    return success();
  }
};

class SubtractOpConversionPattern
    : public OpConversionPattern<ttir::SubtractOp> {
  using OpConversionPattern<ttir::SubtractOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::SubtractOp srcOp, ttir::SubtractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType lhsType =
        mlir::cast<RankedTensorType>(adaptor.getInputs().front().getType());
    RankedTensorType rhsType =
        mlir::cast<RankedTensorType>(adaptor.getInputs().back().getType());

    if (lhsType.getShape() == rhsType.getShape()) {
      rewriter.replaceOpWithNewOp<linalg::SubOp>(
          srcOp, adaptor.getInputs(), adaptor.getOutputs(), srcOp->getAttrs());

      // Broadcast for rhs operand require the operation to be commutative to
      // allow switching the order of operands. To allow this conversion, the
      // following conversion is applied to SubtractOp: subtractOp(lhs,rhs) ->
      // addOp(lhs, negOp(rhs))

    } else {
      auto negEmptyOp = rewriter.create<tensor::EmptyOp>(
          srcOp.getLoc(), rhsType.getShape(), rhsType.getElementType());
      auto negOp = rewriter.create<linalg::NegFOp>(
          srcOp.getLoc(), ValueRange{adaptor.getInputs().back()},
          ValueRange{negEmptyOp}, srcOp->getAttrs());

      rewriter.replaceOpWithNewOp<linalg::AddOp>(
          srcOp,
          ValueRange{adaptor.getInputs().front(), negOp.getResults().front()},
          adaptor.getOutputs(), srcOp->getAttrs());
    }

    return success();
  }
};

} // namespace

namespace mlir::tt {

void populateTTIRToLinAlgPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  // clang-format off
  // ANCHOR: op_rewriter_pattern_set
  patterns
      .add<
        // TensorEmptyConversionPattern,
        //    ToLayoutOpConversionPattern,
        //    ElementwiseOpConversionPattern<ttir::AbsOp, ttnn::AbsOp>,
           ElementwiseOpConversionPattern<ttir::AddOp, linalg::AddOp>,
        //    ElementwiseOpConversionPattern<ttir::CbrtOp, ttnn::CbrtOp>,
        //    ElementwiseOpConversionPattern<ttir::FloorOp, ttnn::FloorOp>,
        //    ElementwiseOpConversionPattern<ttir::IsFiniteOp, ttnn::IsFiniteOp>,
        //    ElementwiseOpConversionPattern<ttir::LogicalAndOp, ttnn::LogicalAndOp>,
        //    ElementwiseOpConversionPattern<ttir::LogicalOrOp, ttnn::LogicalOrOp>,
        //    ElementwiseOpConversionPattern<ttir::LogicalNotOp, ttnn::LogicalNotOp>,
        //    ElementwiseOpConversionPattern<ttir::LogicalXorOp, ttnn::LogicalXorOp>,
           ElementwiseOpConversionPattern<ttir::MultiplyOp, linalg::MulOp>,
        //    ElementwiseOpConversionPattern<ttir::EqualOp, ttnn::EqualOp>,
        //    ElementwiseOpConversionPattern<ttir::NotEqualOp, ttnn::NotEqualOp>,
        //    ElementwiseOpConversionPattern<ttir::GreaterEqualOp, ttnn::GreaterEqualOp>,
        //    ElementwiseOpConversionPattern<ttir::GreaterThanOp, ttnn::GreaterThanOp>,
        //    ElementwiseOpConversionPattern<ttir::LessEqualOp, ttnn::LessEqualOp>,
        //    ElementwiseOpConversionPattern<ttir::LessThanOp, ttnn::LessThanOp>,
        //    ElementwiseOpConversionPattern<ttir::MaximumOp, ttnn::MaximumOp>,
        //    ElementwiseOpConversionPattern<ttir::MinimumOp, ttnn::MinimumOp>,
        //    ElementwiseOpConversionPattern<ttir::NegOp, ttnn::NegOp>,
        //    ElementwiseOpConversionPattern<ttir::ReluOp, ttnn::ReluOp>,
        //    ElementwiseOpConversionPattern<ttir::GeluOp, ttnn::GeluOp>,
        //    ElementwiseOpConversionPattern<ttir::SqrtOp, ttnn::SqrtOp>,
        //    ElementwiseOpConversionPattern<ttir::RsqrtOp, ttnn::RsqrtOp>,
        //    ElementwiseOpConversionPattern<ttir::SignOp, ttnn::SignOp>,
        //    ElementwiseOpConversionPattern<ttir::SigmoidOp, ttnn::SigmoidOp>,
        //    ElementwiseOpConversionPattern<ttir::Log1pOp, ttnn::Log1pOp>,
        //    ElementwiseOpConversionPattern<ttir::ReciprocalOp, ttnn::ReciprocalOp>,
        //    ElementwiseOpConversionPattern<ttir::ExpOp, ttnn::ExpOp>,
        //    ElementwiseOpConversionPattern<ttir::LogOp, ttnn::LogOp>,
        //    ElementwiseOpConversionPattern<ttir::DivOp, ttnn::DivOp>,
        //    ElementwiseOpConversionPattern<ttir::CeilOp, ttnn::CeilOp>,
        //    ElementwiseOpConversionPattern<ttir::SinOp, ttnn::SinOp>,
        //    ElementwiseOpConversionPattern<ttir::CosOp, ttnn::CosOp>,
        //    ElementwiseOpConversionPattern<ttir::Expm1Op, ttnn::Expm1Op>,
        //    ElementwiseOpConversionPattern<ttir::RemainderOp, ttnn::RemainderOp>,
        //    ElementwiseOpConversionPattern<ttir::WhereOp, ttnn::WhereOp>,
        //    ElementwiseUnaryWithFloatParameterOpConversionPattern<ttir::LeakyReluOp, ttnn::LeakyReluOp>,
        //    ReductionOpConversionPattern<ttir::SumOp, ttnn::SumOp>,
        //    ReductionOpConversionPattern<ttir::MeanOp, ttnn::MeanOp>,
        //    ReductionOpConversionPattern<ttir::MaxOp, ttnn::MaxOp>,
        //    BroadcastOpConversionPattern,
        //    EmbeddingOpConversionPattern,
        //    SoftmaxOpConversionPattern,
        //    TransposeOpConversionPattern,
        //    TypecastOpConversionPattern,
        //    ClampOpConversionPattern,
        //    ConcatOpConversionPattern,
        //    ReshapeOpConversionPattern,
        //    SliceOpConversionPattern,
        //    SqueezeOpConversionPattern,
        //    UnsqueezeOpConversionPattern,
        //    ConstantOpConversionPattern,
        //    MatmulOpConversionPattern,
        //    Conv2dOpConversionPattern,
        //    MaxPool2dOpConversionPattern,
           SubtractOpConversionPattern
        //    AllGatherOpConversionPattern
           >(typeConverter, ctx);
  // ANCHOR_END: op_rewriter_pattern_set
  // clang-format on
}

} // namespace mlir::tt
