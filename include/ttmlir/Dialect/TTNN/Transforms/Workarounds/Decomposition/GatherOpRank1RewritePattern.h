// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GATHEROPRANK1REWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GATHEROPRANK1REWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal's gather requires its input and index tensors to be rank >= 2. This
// workaround handles rank-1 operands by unsqueezing a leading unit dimension on
// the input and index, performing the gather on the resulting rank-2 tensors,
// and reshaping the result back to the original rank-1 shape.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/45155
class GatherOpRank1RewritePattern : public OpRewritePattern<ttnn::GatherOp> {
public:
  using OpRewritePattern<ttnn::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::GatherOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GATHEROPRANK1REWRITEPATTERN_H
