// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses Scaled Dot Product Attention from component ops into a single
// ttir.scaled_dot_product_attention op. The TTIR-level matcher targets the
// canonical mathematical form only; layout/precision recovery hacks belong
// to the TTNN-level matcher (which remains as a fallback).
//
// Matches:  matmul(typecast?(softmax(typecast?(score_chain))), V)
//   where score_chain carries Q·Kᵀ (K transposed via ttir.transpose or a
//   last-two-dims ttir.permute, the form dot_general decomposition produces)
//   plus an optional additive mask, in one of these forms:
//     - linear([scale_op?](Q), transpose([scale_op?](K)), bias = mask)
//     - add([scale_op]([matmul]([scale_op?](Q), transpose([scale_op?](K)))),
//     mask)
//     - [scale_op]([matmul]([scale_op?](Q), transpose([scale_op?](K))))   // no
//     mask
//   and scale_op is multiply-or-divide with a `ttir.full` constant.
//
//   The masked form normally arrives as a `ttir.linear`: an earlier sub-phase
//   of TTIRFusing (MatmulWithBiasFusionPattern) folds `add(matmul, mask)` into
//   `linear(Q, Kᵀ, bias = mask)` before this pattern runs. The `add` form is
//   only reached when that fold declines (e.g. a post-scale multiply sits
//   between the matmul and the add, which it does not look through).
//
// Produces: ttir.scaled_dot_product_attention(Q, K, V, mask?, scale?)
//
// Pre/post scaling is supported in three positions: on Q (pre-matmul),
// on K (either side of the transpose), and on the score tensor (post-matmul).
// Double-scaling (both pre- and post-) is rejected as ambiguous.
//
// GQA: a head-dim ttir.repeat_interleave that expands K/V from Hkv to Hq heads
// (through an optional typecast) is peeled from both K and V so the un-expanded
// tensors feed the op, which handles Hkv < Hq natively.
//
// Attention sink ("softmax padding column"): a sink logit concat'd as an extra
// score column before softmax and sliced off after is recognised — the trailing
// slice and the concat are peeled, and the sink (broadcast back to [1, Hq, 1,
// 1]) is fed to the op's attention_sink operand.
//
class SDPAFusingPattern : public mlir::OpRewritePattern<MatmulOp> {
public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;
};

// Peels a GQA head-expansion off the K/V operands of an existing
// ttir.scaled_dot_product_attention. SDPA handles Hkv < Hq natively, so a
// head-dim ttir.repeat_interleave (or HF repeat_kv's unsqueeze/broadcast/reshape
// form) that expands K/V from Hkv to Hq heads before the op is redundant; left
// in place it materialises an Hq-head copy of the KV cache every decode step
// (a ttnn.repeat_interleave per K and V, per layer). SDPAFusingPattern already
// does this peel while fusing the decomposed matmul+softmax form; this pattern
// covers frontends that emit the atomic op directly, where no matmul is left to
// match.
class SDPAGroupedQueryPeelPattern
    : public mlir::OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ScaledDotProductAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H
