// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_TOPKROUTERGPTDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_TOPKROUTERGPTDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which strips the k_padded hardware alignment constraint from the
// outputs of TopKRouterGptOp. tt-metal requires the output dim to be a
// multiple of 8 (k_padded = round_up(k, 8)), so the op emits [B, k_padded]
// tensors. This pattern inserts ttnn::SliceStaticOp after the op to trim both
// outputs back to [B, k], making the hardware detail transparent to downstream
// consumers. Only fires when k % 8 != 0 (i.e. k_padded != k).
class TopKRouterGptDecompositionRewritePattern
    : public OpRewritePattern<ttnn::TopKRouterGptOp> {
public:
  using OpRewritePattern<ttnn::TopKRouterGptOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::TopKRouterGptOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_TOPKROUTERGPTDECOMPOSITIONREWRITEPATTERN_H
