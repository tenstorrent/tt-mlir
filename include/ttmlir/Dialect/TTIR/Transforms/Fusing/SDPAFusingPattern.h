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
//   where score_chain is:
//     [add]([scale_op]([matmul]([scale_op?](Q), transpose([scale_op?](K)))), mask)
//   and scale_op is multiply-or-divide with a `ttir.full` constant.
//
// Produces: ttir.scaled_dot_product_attention(Q, K, V, mask?, scale?)
//
// Pre/post scaling is supported in three positions: on Q (pre-matmul),
// on K (before the transpose), and on the score tensor (post-matmul).
// Double-scaling (both pre- and post-) is rejected as ambiguous.
//
// Out of scope (intentionally — handled by the TTNN matcher today):
//   - generic typecast look-through, NaN-safety slice/concat/where
//   - LinearOp form for the score matmul
//   - repeat_interleave-based GQA expansion (SDPA handles Hkv < Hq natively)
//   - 3D Q/K/V (must be rank 4)
//   - attention_sink, sliding_window_size
class SDPAFusingPattern : public mlir::OpRewritePattern<MatmulOp> {
public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_SDPAFUSINGPATTERN_H
