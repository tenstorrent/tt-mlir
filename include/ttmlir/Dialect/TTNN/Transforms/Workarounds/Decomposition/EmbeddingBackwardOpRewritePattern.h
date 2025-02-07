// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGBACKWARDOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGBACKWARDOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal can not perform embedding backward op for index tensor which is not
// padded on the TILE_WIDTH.
// https://github.com/tenstorrent/tt-metal/issues/17714
// This workaround adds padding to the index tensor to make it padded on the
// TILE_WIDTH.
class EmbeddingBackwardOpRewritePattern
    : public OpRewritePattern<ttnn::EmbeddingBackwardOp> {
public:
  using OpRewritePattern<ttnn::EmbeddingBackwardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::EmbeddingBackwardOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif
