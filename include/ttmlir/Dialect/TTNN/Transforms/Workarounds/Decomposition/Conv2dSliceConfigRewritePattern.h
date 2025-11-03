// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DSLICECONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DSLICECONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Set Conv2dSliceConfig to L1Full with 0 slices as a workaround.
// This configuration is a safe default that avoids potential issues
// with slicing while ensuring compatibility with existing models.
// Metal issue tracking this workaround:
// https://github.com/tenstorrent/tt-metal/issues/29981
class Conv2dSliceConfigRewritePattern
    : public mlir::OpRewritePattern<ttnn::Conv2dOp> {
public:
  using mlir::OpRewritePattern<ttnn::Conv2dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttnn::Conv2dOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DSLICECONFIGREWRITEPATTERN_H
