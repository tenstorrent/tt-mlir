// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTEROPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTEROPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ReduceScatterOp in TTNN currently does not support tensors with rank < 4
// correctly. As a temporary workaround, we insert reshape ops front and back
// to make the tensor as four dimensional tensor.
// Related tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/15010
class TTNNReduceScatterWorkarounds
    : public OpRewritePattern<ttnn::ReduceScatterOp> {
public:
  using OpRewritePattern<ttnn::ReduceScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ReduceScatterOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REDUCESCATTEROPEREWRITEPATTERN_H