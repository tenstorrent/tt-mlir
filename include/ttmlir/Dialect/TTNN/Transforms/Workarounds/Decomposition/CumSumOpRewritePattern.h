// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// This workaround addresses the tt-metal issue:
// https://github.com/tenstorrent/tt-metal/issues/14549
//
// TODO(mmanzoor): Remove this workaround once these Metal issues are fixed
// (tracked by https://github.com/tenstorrent/tt-mlir/issues/1624).
class CumSumOpRewritePattern : public OpRewritePattern<ttnn::MorehCumSumOp> {
public:
  using OpRewritePattern<ttnn::MorehCumSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPSREWRITEPATTERN_H
