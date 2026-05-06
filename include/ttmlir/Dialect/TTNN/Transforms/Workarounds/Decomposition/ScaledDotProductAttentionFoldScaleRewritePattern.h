// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONFOLDSCALEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONFOLDSCALEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Folds residual scalar multiplies on the Q and/or K operands of a
// ttnn.scaled_dot_product_attention into the SDPA's `scale` attribute.
// Looks through optional one-use ttnn.typecast and ttnn.permute ops between
// the multiply and the SDPA. Mathematically equivalent: SDPA(c*Q, K, V, s) ==
// SDPA(Q, K, V, s*c). Combines existing scale with detected scalars.
class ScaledDotProductAttentionFoldScaleRewritePattern
    : public OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONFOLDSCALEREWRITEPATTERN_H
