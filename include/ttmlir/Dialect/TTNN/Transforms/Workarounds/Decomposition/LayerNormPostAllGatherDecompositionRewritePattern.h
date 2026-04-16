// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPOSTALLGATHERDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPOSTALLGATHERDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Decomposes LayerNormPostAllGatherOp into primitive TTNN ops when the input
// last dimension is not tile-aligned (W % TILE_WIDTH != 0).
//
// Background: the tt-metal layer_norm_post_all_gather kernel computes its
// normalization scaler as winv = 1 / (padded_shape[-1] * num_devices) instead
// of 1 / (logical_shape[-1] * num_devices).  When W is not a multiple of
// TILE_WIDTH the padded shape is larger than the logical shape, so the mean and
// variance are computed over phantom zero-padded elements and the output is
// incorrect.
//
// This decomposition manually extracts sum(x²) and sum(x) from the gathered
// stats tensor (using the known stats format produced by pre_all_gather), then
// normalizes the input with the correct count N = W * num_devices:
//
//   total_sum_x2 = Σ_k stats[..., k*64   : k*64+1 ]   shape [..., 1]
//   total_sum_x  = Σ_k stats[..., k*64+32: k*64+33]   shape [..., 1]
//   winv   = 1 / (W * num_devices)
//   E[x²]  = total_sum_x2 * winv
//   E[x]   = total_sum_x  * winv
//   var    = E[x²] - E[x] * E[x]
//   rstd   = rsqrt(var + epsilon)
//   result = (input - E[x]) * rstd [* weight] [+ bias]
//
// When W is tile-aligned the kernel is correct and this pattern does not fire.
// Metal issue tracking this workaround:
// https://github.com/tenstorrent/tt-metal/issues/41553
class LayerNormPostAllGatherDecompositionRewritePattern
    : public OpRewritePattern<ttnn::LayerNormPostAllGatherOp> {
public:
  using OpRewritePattern<ttnn::LayerNormPostAllGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::LayerNormPostAllGatherOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LAYERNORMPOSTALLGATHERDECOMPOSITIONREWRITEPATTERN_H
