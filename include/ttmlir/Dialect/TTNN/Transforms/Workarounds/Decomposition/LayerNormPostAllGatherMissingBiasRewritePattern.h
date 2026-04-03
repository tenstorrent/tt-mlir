// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPOSTALLGATHERMISSINGBIASREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPOSTALLGATHERMISSINGBIASREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround for the tt-metal layer_norm_post_all_gather constraint:
// bias (beta) must be present whenever weight (gamma) is present
// (is_layernorm == has_beta). When a LayerNormPostAllGatherOp has weight but no
// bias, this pattern synthesizes a zero bias tensor of the same type as the
// weight so that the backend can emit a valid layer_norm_post_all_gather call.
// Metal issue: https://github.com/tenstorrent/tt-metal/issues/41378
class LayerNormPostAllGatherMissingBiasRewritePattern
    : public OpRewritePattern<ttnn::LayerNormPostAllGatherOp> {
public:
  using OpRewritePattern<ttnn::LayerNormPostAllGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::LayerNormPostAllGatherOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPOSTALLGATHERMISSINGBIASREWRITEPATTERN_H
