// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

class FullOpWithNanConstantRewritePattern
    : public OpRewritePattern<ttnn::FullOp> {
  using OpRewritePattern<ttnn::FullOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(ttnn::FullOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition
