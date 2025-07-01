// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_UPSAMPLEOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
// UpsampleOp in `bilinear` mode expects the input to be in `HeightSharded`
// memory layout. This pattern converts the input tensor to the target memory
// layout, provided that the input tensor is already in a `RowMajor` layout and
// has `BFloat16` data type, and that the channel axis is a multiple of the tile
// width.
class UpsampleOpBilinearShardingRewritePattern
    : public OpRewritePattern<ttnn::UpsampleOp> {
public:
  UpsampleOpBilinearShardingRewritePattern(MLIRContext *ctx)
      : OpRewritePattern<ttnn::UpsampleOp>(ctx, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ttnn::UpsampleOp srcOp,
                                PatternRewriter &rewriter) const override;
};

// This pattern is a prerequisite (hence the `2` benefit) for the
// `UpsampleOpBilinearShardingRewritePattern`. `Channel` axis of the input has
// to be a multiple of the tile width.
class UpsampleOpBilinearPaddingRewritePattern
    : public OpRewritePattern<ttnn::UpsampleOp> {
public:
  UpsampleOpBilinearPaddingRewritePattern(MLIRContext *ctx)
      : OpRewritePattern<ttnn::UpsampleOp>(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(ttnn::UpsampleOp srcOp,
                                PatternRewriter &rewriter) const override;
};

// UpsampleOp expects the input to be in `RowMajor` layout and has `BFloat16`
// data type. This pattern converts the input tensor to the target memory
// configuration.
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
