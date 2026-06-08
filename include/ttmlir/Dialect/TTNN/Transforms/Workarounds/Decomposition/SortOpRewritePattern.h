// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SORTOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SORTOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Works around tt-metal #46331: `ttnn.sort` compares keys in bf16, so integer
// keys > 256 collide and the returned indices are wrong (the values are
// fine).
//
// For an integer sort whose indices are used, recompute the indices by
// rank-by-comparison in f32 (exact for |x| <= 2^24), tiled to bound memory:
//   rank[i]    = #{j: x[j] < x[i]} + #{j: x[j]==x[i] and j<i}   (stable)
//   argsort[k] = sum_i i * (rank[i] == k)                       (inverse)
// Values still come from `ttnn.sort`.
class SortOpRewritePattern : public OpRewritePattern<ttnn::SortOp> {
public:
  using OpRewritePattern<ttnn::SortOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SortOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SORTOPREWRITEPATTERN_H
