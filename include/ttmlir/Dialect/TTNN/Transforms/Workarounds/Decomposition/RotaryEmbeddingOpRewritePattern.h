// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ROTARYEMBEDDINGOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ROTARYEMBEDDINGOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
class RotaryEmbeddingOpRewritePattern
    : public OpRewritePattern<RotaryEmbeddingOp> {
public:
  using OpRewritePattern<RotaryEmbeddingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RotaryEmbeddingOp srcOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_PASSES_TTNNFUSINGOPREWRITEPATTERN_H
