// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DSLICECONFIGULTRASPECIFICREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DSLICECONFIGULTRASPECIFICREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

  // TL;DR: This is a temporary workaround to avoid L1 OOM issues with particular convolutions in SpeechT5 vocoder model (particularly, conv_post module).
  // Going from L1Full memory config to DramWidth avoids L1 OOM issues.
class Conv2dSliceConfigUltraSpecificRewritePattern
    : public mlir::OpRewritePattern<ttnn::Conv2dOp> {
public:
  using mlir::OpRewritePattern<ttnn::Conv2dOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttnn::Conv2dOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DSLICECONFIGULTRASPECIFICREWRITEPATTERN_H
