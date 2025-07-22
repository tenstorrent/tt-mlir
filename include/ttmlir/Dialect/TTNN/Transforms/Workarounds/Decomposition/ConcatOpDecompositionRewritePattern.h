// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TTNN does not support concatenating more than 50 input tensors.
// https://github.com/tenstorrent/tt-metal/issues/22845
// This workaround decomposes the concatenation operation into multiple ops if
// the number of input tensors exceeds 50.
class ConcatOpDecompositionRewritePattern
    : public OpRewritePattern<ttnn::ConcatOp> {
public:
  using OpRewritePattern<ttnn::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConcatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPDECOMPOSITIONREWRITEPATTERN_H
