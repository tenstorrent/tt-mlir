// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

class Conv2dOpRewritePattern : public OpRewritePattern<ttnn::Conv2dOp> {
public:
  using OpRewritePattern<ttnn::Conv2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::Conv2dOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV2DOPREWRITEPATTERN_H
