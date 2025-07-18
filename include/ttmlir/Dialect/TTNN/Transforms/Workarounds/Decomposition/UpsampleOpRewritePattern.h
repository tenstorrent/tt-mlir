// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// `Channel` axis of the input has to be a multiple of the tile width.
class UpsampleOpBilinearPaddingRewritePattern
    : public OpRewritePattern<ttnn::UpsampleOp> {
public:
  using OpRewritePattern<ttnn::UpsampleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::UpsampleOp srcOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H
