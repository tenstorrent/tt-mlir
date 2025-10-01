// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MULTIPLYTYPECASTREMOVALREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_MULTIPLYTYPECASTREMOVALREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// This pattern removes unnecessary typecast operations around multiply ops.
// It matches multiply ops where:
// 1. Both operands are typecasts from bf16 to f32
// 2. The multiply op has only one user
// 3. That user is a typecast from f32 to bf16
// In such cases, the pattern removes all three typecasts and performs the
// multiply directly in bf16.
// TODO: Remove this workaround once the underlying issue is addressed.
class MultiplyTypecastRemovalRewritePattern
    : public OpRewritePattern<ttnn::MultiplyOp> {
public:
  using OpRewritePattern<ttnn::MultiplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MultiplyOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif
