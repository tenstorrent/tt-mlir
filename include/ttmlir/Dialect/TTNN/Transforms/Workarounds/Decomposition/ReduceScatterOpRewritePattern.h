// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTEROPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTEROPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ReduceScatterOp in TTNN now supports 4d input tensors, but it seems that there
// are some numerical accuracy issues when the input tensor has rank < 4.
// Related tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/39953
class TTNNReduceScatterWorkarounds
    : public OpRewritePattern<ttnn::ReduceScatterOp> {
public:
  using OpRewritePattern<ttnn::ReduceScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ReduceScatterOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTEROPREWRITEPATTERN_H
