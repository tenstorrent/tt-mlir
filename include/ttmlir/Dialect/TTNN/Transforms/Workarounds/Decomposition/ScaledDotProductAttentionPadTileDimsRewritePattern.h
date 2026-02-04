// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONPADTILEDIMSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONPADTILEDIMSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which adds padding to ScaledDotProductAttention query, key, and
// value tensors to make:
// - sequence dimensions (dim -2) multiples of TILE_HEIGHT (only when attention
//   mask is present) - Metal issue reference:
//   https://github.com/tenstorrent/tt-metal/issues/32502
// - head dimensions (dim -1) multiples of TILE_WIDTH (always, regardless of
//   attention mask) - Metal issue reference:
//   https://github.com/tenstorrent/tt-metal/issues/33434
// When attention mask is present and sequence padding is needed, the mask is
// also padded accordingly with -infinity values.
// After the operation, the result is sliced back to the original shape.

class ScaledDotProductAttentionPadTileDimsRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ttnn::ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONPADTILEDIMSREWRITEPATTERN_H
