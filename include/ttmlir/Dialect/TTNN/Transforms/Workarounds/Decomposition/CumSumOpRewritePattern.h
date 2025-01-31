// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal can not perform cumsum op for input tensor with rank < 4.
// https://github.com/tenstorrent/tt-metal/issues/17241
// This workaround unsqueeze the input tensor to 4D tennsor (if required) and
// reshape it back to original shape after performing the cumsum op.
class CumSumOpRewritePattern : public OpRewritePattern<ttnn::MorehCumSumOp> {
public:
  using OpRewritePattern<ttnn::MorehCumSumOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CUMSUMOPREWRITEPATTERN_H
