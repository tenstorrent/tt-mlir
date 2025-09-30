// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_UTILS_D2MTILEOPREWRITER_H
#define TTMLIR_CONVERSION_UTILS_D2MTILEOPREWRITER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::d2m {

template <typename SourceOp, typename D2MTileOp>
class UnaryTileOpRewriter : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<D2MTileOp>(op, op.getResult().getType(),
                                           operands[0]);
    return success();
  }
};

template <typename SourceOp, typename D2MTileOp>
class BinaryTileOpRewriter : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<D2MTileOp>(op, op.getResult().getType(),
                                           operands[0], operands[1]);
    return success();
  }
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_CONVERSION_UTILS_D2MTILEOPREWRITER_H
