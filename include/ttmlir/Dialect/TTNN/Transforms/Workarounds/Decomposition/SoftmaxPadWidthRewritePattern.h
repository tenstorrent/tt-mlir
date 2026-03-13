// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SOFTMAXPADWIDTHREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SOFTMAXPADWIDTHREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround for tt-metal issue where the attention-optimized softmax kernel's
// large-kernel fallback path uses a hardcoded circular buffer length of 80
// tiles, which must be divisible by block_size = find_max_divisor(Wt, N).
// When Wt (width in tiles) yields a block_size of 3 or 6, 80 % block_size != 0
// triggers a TT_FATAL.
//
// This pattern pads the softmax width dimension so the tile count produces a
// safe block_size (one that divides 80). After softmax, the result is sliced
// back to the original shape.
//
// Applies only to rank-4 tensors with softmax on the last dimension, matching
// the conditions under which the AttentionOptimized factory is selected.

// TT-Metal https://github.com/tenstorrent/tt-metal/issues/38626

class SoftmaxPadWidthRewritePattern : public OpRewritePattern<ttnn::SoftmaxOp> {
public:
  using OpRewritePattern<ttnn::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SoftmaxOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SOFTMAXPADWIDTHREWRITEPATTERN_H
