// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGBACKWARDOPUNSQUEEZEINDICESREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGBACKWARDOPUNSQUEEZEINDICESREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

class EmbeddingBackwardOpUnsqueezeIndicesRewritePattern
    : public OpRewritePattern<ttnn::EmbeddingBackwardOp> {
public:
  using OpRewritePattern<ttnn::EmbeddingBackwardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::EmbeddingBackwardOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGBACKWARDOPUNSQUEEZEINDICESREWRITEPATTERN_H
