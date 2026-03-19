// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ARGMAXOPDIMREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ARGMAXOPDIMREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal supports ArgMax op only when reducing on the last dimension.
// This workaround permutes the input tensor so that the reduction dimension
// becomes the last dimension, performs ArgMax on it, and then permutes the
// output back to the original dimension order.
// Metal issue to add support for ArgMax on arbitrary dimensions:
// https://github.com/tenstorrent/tt-metal/issues/40218
class ArgMaxOpDimRewritePattern : public OpRewritePattern<ttnn::ArgMaxOp> {
public:
  using OpRewritePattern<ttnn::ArgMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ArgMaxOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ARGMAXOPDIMREWRITEPATTERN_H
