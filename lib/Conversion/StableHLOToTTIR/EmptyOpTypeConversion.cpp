// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/EmptyOpTypeConversion.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::tt;

namespace {
class EmptyOpTypeConversionPattern
    : public OpConversionPattern<mlir::tt::ttir::EmptyOp> {

  using OpConversionPattern<mlir::tt::ttir::EmptyOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::tt::ttir::EmptyOp srcOp,
                  mlir::tt::ttir::EmptyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType inputType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getType()));
    rewriter.replaceOpWithNewOp<mlir::tt::ttir::EmptyOp>(
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
