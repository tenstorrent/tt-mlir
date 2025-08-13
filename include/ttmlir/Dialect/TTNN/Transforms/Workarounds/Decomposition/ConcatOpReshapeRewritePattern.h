// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPRESHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPRESHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// ConcatOp crashes if the inputs are 1 dimensional tensors.
// tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/22541

// This workaround adds one additional dimension to all the input tensor (using
// reshape) to make input tensors 2 dimensional and reshaped back after
// concatenation.
class ConcatOpReshapeRewritePattern : public OpRewritePattern<ttnn::ConcatOp> {
public:
  using OpRewritePattern<ttnn::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConcatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATOPRESHAPEREWRITEPATTERN_H
