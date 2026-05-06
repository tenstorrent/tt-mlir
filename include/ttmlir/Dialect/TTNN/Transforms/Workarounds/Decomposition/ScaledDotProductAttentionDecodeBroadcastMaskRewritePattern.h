// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEBROADCASTMASKREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEBROADCASTMASKREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which materializes the heads dim (dim 2) of the decode SDPA
// attention mask via an explicit repeat when it was emitted as 1.
//
// Decode mask layout is [1|B, 1, 1|Hq, Sk]. The kernel:
//   - natively broadcasts the batch dim (dim 0): mask[0] may be 1 for all B,
//   - does not broadcast the heads dim (dim 2): mask[2] must equal Hq.
// Only the heads-dim case needs this rewrite; batch broadcast is handled by
// the kernel directly.
//
// Tracking issue: https://github.com/tenstorrent/tt-metal/issues/39946

class ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern
    : public OpRewritePattern<ttnn::ScaledDotProductAttentionDecodeOp> {
public:
  using OpRewritePattern<
      ttnn::ScaledDotProductAttentionDecodeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ScaledDotProductAttentionDecodeOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONDECODEBROADCASTMASKREWRITEPATTERN_H
