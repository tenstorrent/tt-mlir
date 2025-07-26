// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LEGALIZEPOOLPADDINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LEGALIZEPOOLPADDINGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// `Channel` axis of the input has to be a multiple of the tile width.
template <typename Pool2dOp>
class LegalizePoolPaddingRewritePattern : public OpRewritePattern<Pool2dOp> {
public:
  using OpRewritePattern<Pool2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Pool2dOp srcOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LEGALIZEPOOLPADDINGREWRITEPATTERN_H
