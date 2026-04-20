// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DISTRIBUTEDLAYERNORMDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DISTRIBUTEDLAYERNORMDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::decomposition {

// Decomposes DistributedLayerNormOp unconditionally into three dedicated TTNN
// ops that implement the distributed layer normalization pipeline:
//   1. layer_norm_pre_all_gather: compute local partial statistics (Welford)
//   2. all_gather: gather statistics across devices along cluster_axis
//   3. layer_norm_post_all_gather: normalize using gathered global statistics
// If residual is present, an explicit add(input, residual) is emitted first
// and norm_input is passed to both pre and post ops.
class DistributedLayerNormDecompositionRewritePattern
    : public OpRewritePattern<ttnn::DistributedLayerNormOp> {
public:
  using OpRewritePattern<ttnn::DistributedLayerNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::DistributedLayerNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DECOMPOSITION_DISTRIBUTEDLAYERNORMDECOMPOSITIONREWRITEPATTERN_H
