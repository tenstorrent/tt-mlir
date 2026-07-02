// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_SDPAERASEREPEATKV_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_SDPAERASEREPEATKV_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn {

// Erases the explicit repeat_interleave (repeat_kv) expansion that a frontend
// emits before scaled_dot_product_attention for grouped-/multi-query attention
// (GQA/MQA). The TTNN SDPA op broadcasts K/V across query-head groups
// internally, so the un-expanded K/V can be fed directly and the repeat is
// redundant.
//
// Matches:
//   K' = repeat_interleave(K, dim=1, repeats=R)   // [B, Hkv, S, D] -> [B, Hq,
//   S, D] V' = repeat_interleave(V, dim=1, repeats=R)
//   scaled_dot_product_attention(Q, K', V', ...)   // Q: [B, Hq, S, D]
//
// Rewrites (in place) to:
//   scaled_dot_product_attention(Q, K, V, ...)     // K/V: [B, Hkv, S, D]
//
// Guards: both K and V must be repeat_interleave on the head dim with the same
// repeat factor R, and Q's head count must equal R * Hkv. This makes the
// rewrite numerically identity-preserving.
class SDPAEraseRepeatKV
    : public mlir::OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  using mlir::OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ScaledDotProductAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_SDPAERASEREPEATKV_H
