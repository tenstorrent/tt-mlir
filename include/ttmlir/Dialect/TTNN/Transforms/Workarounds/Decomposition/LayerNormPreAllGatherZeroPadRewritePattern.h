// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPREALLGATHERZEROPADREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPREALLGATHERZEROPADREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround for the tt-metal layer_norm_pre_all_gather kernel contract:
// the kernel reads the full tile-padded width when computing sum(x) and
// sum(x^2). When the input width W is not a multiple of TILE_WIDTH (32), the
// tile-padding area (positions W..ceil(W/32)*32-1) may contain non-zero
// garbage, corrupting sum(x) and sum(x^2).
//
// Fix: pad with zeros to fill the tile area so the kernel accumulates zeros
// over the padding region, ensuring correct sum(x) and sum(x^2) values.
// No slice back to the original width is required because the stats output
// shape is independent of the input width.
//
// Note: the post_all_gather kernel has a separate issue where it computes
// winv using padded_shape instead of logical_shape (see issue #41553).
// That is handled by LayerNormPostAllGatherDecompositionRewritePattern.
//
// Rewrites:
//   layer_norm_pre_all_gather(input[..., W], ...)      # W % 32 != 0
// into:
//   padded = ttnn.pad(input, [..., 0, pad_high], value=0.0)
//   layer_norm_pre_all_gather(padded[..., ceil(W/32)*32], ...)
//
// Metal issue tracking this workaround:
// https://github.com/tenstorrent/tt-metal/issues/41465
class LayerNormPreAllGatherZeroPadRewritePattern
    : public OpRewritePattern<ttnn::LayerNormPreAllGatherOp> {
public:
  using OpRewritePattern<ttnn::LayerNormPreAllGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::LayerNormPreAllGatherOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPREALLGATHERZEROPADREWRITEPATTERN_H
