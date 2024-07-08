// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_LIB_CONVERSION_TTNNTOEMITC_POPULATEPATTERNS_H
#define TTMLIR_LIB_CONVERSION_TTNNTOEMITC_POPULATEPATTERNS_H

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

template <typename SrcOp, typename Adaptor = typename SrcOp::Adaptor>
class DefaultOpConversionPattern : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;

public:
  // Default op conversion pattern, used to convert most ops
  //
  DefaultOpConversionPattern(MLIRContext *ctx)
      : OpConversionPattern<SrcOp>(ctx) {}

  DefaultOpConversionPattern(const TypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<SrcOp>(typeConverter, context, benefit) {}

  // Coverts op name by removing the dialect prefix ("ttnn.") and replacing with
  // namespace prefix ("ttnn::")
  //
  std::string convertOpName(SrcOp op) const {
    auto name = op.getOperationName();
    if (name.starts_with("ttnn.")) {
      return "ttnn::" + name.drop_front(5).str();
    }
    return "ttnn::" + name.str();
  }

  LogicalResult
  matchAndRewrite(SrcOp srcOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Attribute, 4> templateArguments;

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

#endif // TTMLIR_LIB_CONVERSION_TTNNTOEMITC_POPULATEPATTERNS_H
