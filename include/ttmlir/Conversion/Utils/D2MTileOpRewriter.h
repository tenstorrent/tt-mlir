// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::d2m {

template <typename SourceOp, typename D2MTileOp, int Arity>
class TileOpRewriter : public OpConversionPattern<SourceOp> {
public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  static_assert(Arity == 1 || Arity == 2);

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto operands = adaptor.getOperands();
    if constexpr (Arity == 1) {
      rewriter.replaceOpWithNewOp<D2MTileOp>(op, op.getResult().getType(),
                                             operands[0]);
      return success();
    } else if constexpr (Arity == 2) {
      rewriter.replaceOpWithNewOp<D2MTileOp>(op, op.getResult().getType(),
                                             operands[0], operands[1]);
      return success();
    }

    return failure();
  }
};

} // namespace mlir::tt::d2m
