// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TTNN ScaledDotProductAttention requires sequence dimensions to be aligned to
// TILE_HEIGHT (32). This workaround pads the query, key, value, and mask
// tensors to ensure alignment, then slices the result back to the original
// sequence length.
class ScaledDotProductAttentionPadQueryRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ttnn::ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONOPREWRITEPATTERN_H
