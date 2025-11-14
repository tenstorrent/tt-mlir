// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONUNSQUEEZEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONUNSQUEEZEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TTNN ScaledDotProductAttention requires 4D inputs. This workaround handles
// SDPA ops with 3D inputs by unsqueezing them to 4D (prepending a dimension of
// size 1), creating the SDPA op with 4D inputs, and squeezing the output back
// to 3D if the original output was 3D.
class ScaledDotProductAttentionUnsqueezeRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ttnn::ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONUNSQUEEZEREWRITEPATTERN_H
