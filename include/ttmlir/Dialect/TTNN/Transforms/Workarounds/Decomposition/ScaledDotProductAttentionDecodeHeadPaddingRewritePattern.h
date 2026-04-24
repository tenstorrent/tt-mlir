// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEHEADPADDINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEHEADPADDINGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround that pads the query num_heads dimension (dim 2) of
// ScaledDotProductAttentionDecodeOp to the next power-of-2 multiple of
// TILE_HEIGHT, and slices the output back to the original num_heads.
//
// tt-metal's sdpa_decode kernel computes
//   MUL_BCAST_GRANULARITY = min(PNHt * Sk_chunk_t, dst_size)
// where PNHt = num_q_heads / TILE_HEIGHT, and asserts this is a power of 2.
// Models such as GLM-4.7 (96 query heads) produce PNHt = 3, which fails.
//
// The pattern pads the query (and attention_sink if present in 2D form, and
// the attention_mask if it has already been broadcast to match num_heads) to
// padded_num_heads = NextPowerOf2(PNHt) * TILE_HEIGHT, creating an op that
// satisfies the tt-metal constraint. The result is sliced back to the original
// num_heads so downstream ops see the correct shape.
//
// When the attention_sink is not yet in rank-2 form the pattern defers,
// allowing ScaledDotProductAttentionDecodeAttentionSinkRewritePattern to
// normalize the sink first.

class ScaledDotProductAttentionDecodeHeadPaddingRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp> {
public:
  using OpRewritePattern<
      ttnn::ScaledDotProductAttentionDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionDecodeOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEHEADPADDINGREWRITEPATTERN_H
