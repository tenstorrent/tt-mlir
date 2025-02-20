// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/EmptyOpTypeConversion.h"

#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace mlir;
using namespace mlir::tt;

namespace {
class EmptyOpTypeConversionPattern
    : public OpConversionPattern<mlir::tensor::EmptyOp> {

  using OpConversionPattern<mlir::tensor::EmptyOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::tensor::EmptyOp srcOp,
                  mlir::tensor::EmptyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getType()));
    rewriter.replaceOpWithNewOp<mlir::tensor::EmptyOp>(
        srcOp, inputType.getShape(), inputType.getElementType(),
        inputType.getEncoding());

    return success();
  }
};
} // namespace

namespace mlir::tt {
void addEmptyOpTypeConversionPattern(MLIRContext *ctx,
                                     RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  patterns.add<EmptyOpTypeConversionPattern>(typeConverter, ctx);
}
} // namespace mlir::tt
