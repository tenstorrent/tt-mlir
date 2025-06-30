// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// TODO (azecevic): Add description!
class UpsampleOpBilinearShardingRewritePattern
    : public OpRewritePattern<ttnn::UpsampleOp> {
public:
  UpsampleOpBilinearShardingRewritePattern(MLIRContext *ctx)
      : OpRewritePattern<ttnn::UpsampleOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ttnn::UpsampleOp srcOp,
                                PatternRewriter &rewriter) const override;
};

class UpsampleOpBilinearPaddingRewritePattern
    : public OpRewritePattern<ttnn::UpsampleOp> {
public:
  UpsampleOpBilinearPaddingRewritePattern(MLIRContext *ctx)
      : OpRewritePattern<ttnn::UpsampleOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(ttnn::UpsampleOp srcOp,
                                PatternRewriter &rewriter) const override;
};

class UpsampleOpLayoutRewritePattern
    : public OpRewritePattern<ttnn::UpsampleOp> {
public:
  UpsampleOpLayoutRewritePattern(MLIRContext *ctx)
      : OpRewritePattern<ttnn::UpsampleOp>(ctx, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(ttnn::UpsampleOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H
