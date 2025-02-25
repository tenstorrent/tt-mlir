// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ARGMAXOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ARGMAXOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal supports ArgMax op for 4D tensors only.
// https://github.com/tenstorrent/tt-metal/issues/18241
// This workaround unsqueeze the input tensor to 4D tennsor (if required) and
// reshape it back to original shape after performing the ArgMax op.
class ArgMaxOpRewritePattern : public OpRewritePattern<ttnn::ArgMaxOp> {
public:
  using OpRewritePattern<ttnn::ArgMaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ArgMaxOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ARGMAXOPREWRITEPATTERN_H
