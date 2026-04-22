// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// When a Conv3dOp has no Conv3dConfig attached, tt-metal's program factory
// falls back to C_in_block = C_in and C_out_block = padded_C_out, which for
// vision-encoder patch embeds (e.g. Qwen3.5-VL: C_in=3, C_out=1152,
// kernel=(2,14,14)) forces the entire tiled weight matrix into L1 and blows
// the per-core circular-buffer budget.
//
// This pattern attaches a Conv3dConfigAttr with conservative block sizes so
// the weight and intermediate buffers fit in L1. It only fires when the
// estimated weight footprint would exceed a safe threshold.
class Conv3dConfigRewritePattern : public OpRewritePattern<ttnn::Conv3dOp> {
public:
  using OpRewritePattern<ttnn::Conv3dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::Conv3dOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DCONFIGREWRITEPATTERN_H
