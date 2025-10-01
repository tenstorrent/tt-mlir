// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_AVGPOOL2DREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_AVGPOOL2DREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
class AvgPool2dRewritePattern : public OpRewritePattern<ttnn::AvgPool2dOp> {
public:
    using OpRewritePattern<ttnn::AvgPool2dOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ttnn::AvgPool2dOp op, PatternRewriter &rewriter) const override;
};


} // namespace mlir::tt::ttnn::workarounds::decomposition



#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_AVGPOOL2DREWRITEPATTERN_H