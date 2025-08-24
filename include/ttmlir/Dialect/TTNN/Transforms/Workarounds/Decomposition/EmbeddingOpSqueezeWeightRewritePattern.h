// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGOPSQUEEZEWEIGHTREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGOPSQUEEZEWEIGHTREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TTNN EmbeddingOp supports at most 4D weight tensor, where all dimensions
// except the last 2 are equal to 1. This pattern rewrites the EmbeddingOp to a
// 4D tensor with the first two dimensions squeezed to 1, and the last two
// dimensions being the original weight tensor dimensions. This is done to
// ensure compatibility with the TTNN EmbeddingOp, which requires a 4D tensor as
// input.
class EmbeddingOpSqueezeWeightRewritePattern
    : public OpRewritePattern<ttnn::EmbeddingOp> {
public:
  using OpRewritePattern<ttnn::EmbeddingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::EmbeddingOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGOPSQUEEZEWEIGHTREWRITEPATTERN_H
