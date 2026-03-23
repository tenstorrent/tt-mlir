// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEATTENTIONSINKREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEATTENTIONSINKREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which reshapes the attention sink from 4D [*, num_heads, 1, 1]
// to the 2D [num_heads, TILE_WIDTH] format expected by tt-metal's SDPA decode.
// The sink is reshaped to [num_heads, 1] and then padded with zeros on the last
// dimension to reach TILE_WIDTH (32).
class ScaledDotProductAttentionDecodeAttentionSinkRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp> {
public:
  using OpRewritePattern<
      ttnn::ScaledDotProductAttentionDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionDecodeOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEATTENTIONSINKREWRITEPATTERN_H
