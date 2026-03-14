// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECKBROADCASTMASKREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECKBROADCASTMASKREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which broadcasts the attention mask num_heads dimension (dim 2)
// for ScaledDotProductAttentionDecode when it is 1 (broadcast) to match the
// query num_heads. tt-metal requires mask[2] == num_heads for decode SDPA and
// does not support implicit broadcasting on that dimension.

class ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp> {
public:
  using OpRewritePattern<
      ttnn::ScaledDotProductAttentionDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionDecodeOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECKBROADCASTMASKREWRITEPATTERN_H
