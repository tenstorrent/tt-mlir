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

    rewriter.replaceOpWithNewOp<LinAlgOpTy>(
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
  patterns.add<ElementwiseOpConversionPattern<ttir::AddOp, linalg::AddOp>,
               ElementwiseOpConversionPattern<ttir::MultiplyOp, linalg::MulOp>,
               SubtractOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
