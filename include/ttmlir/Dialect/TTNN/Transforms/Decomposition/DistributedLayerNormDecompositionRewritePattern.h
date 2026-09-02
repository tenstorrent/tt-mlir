// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DISTRIBUTEDLAYERNORMDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DISTRIBUTEDLAYERNORMDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::decomposition {

// Lowers DistributedLayerNormOp in one of two ways:
//   1. DiT fused kernel form: skip the sandwich when the op is eligible
//      (bf16 or f32, weight+bias, no residual, tile-aligned last dim).
//      Rank-3 `[1,N,H]` is reshaped to `[1,1,N,H]` and 1D affine params
//      to `[1,H]`. f32 is typecast to TILE bf16 around the fused op.
//      Returning failure on an already-canonical eligible bf16 op
//      leaves it for serialization.
//   2. Sandwich: layer_norm_pre_all_gather + all_gather +
//      layer_norm_post_all_gather. Residual, unaffine, and ineligible shapes
//      take this path. Rank < 4 is first reshaped to 1x1xHxW.
class DistributedLayerNormDecompositionRewritePattern
    : public OpRewritePattern<ttnn::DistributedLayerNormOp> {
public:
  using OpRewritePattern<ttnn::DistributedLayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::DistributedLayerNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DISTRIBUTEDLAYERNORMDECOMPOSITIONREWRITEPATTERN_H
