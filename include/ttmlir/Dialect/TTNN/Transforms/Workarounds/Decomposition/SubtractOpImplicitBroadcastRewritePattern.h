// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SUBTRACTOPIMPLICITBROADCASTREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SUBTRACTOPIMPLICITBROADCASTREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Broadcast for rhs operand require the operation to be commutative to
// allow switching the order of operands. To allow this conversion, the
// following conversion is applied to SubtractOp: subtractOp(lhs,rhs) ->
// addOp(lhs, negOp(rhs)).
// Remove once the following issue is resolved:
// https://github.com/tenstorrent/tt-metal/issues/24635.
class SubtractOpImplicitBroadcastRewritePattern
    : public OpRewritePattern<ttnn::SubtractOp> {
public:
  using OpRewritePattern<ttnn::SubtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SubtractOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SUBTRACTOPIMPLICITBROADCASTREWRITEPATTERN_H
