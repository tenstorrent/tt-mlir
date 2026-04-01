// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMDECOMPOSITIONREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMDECOMPOSITIONREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround that decomposes DistributedRMSNormOp into primitive TTNN ops
// when the input shape is not (1,1,32,M). The fused_rms_minimal kernel only
// supports (1,1,32,M) inputs. For other shapes, the op is decomposed into:
//   1. Pre all-gather: multiply(x,x) -> mean(x_sq, dim=-1) (local E(x^2))
//   2. All-gather: gather local stats across devices
//   3. Post all-gather: mean -> add(eps) -> rsqrt -> multiply(x, inv_rms)
//      -> multiply(weight)
// This decomposition mimics the structure of tt-metal's
// rms_norm_pre_all_gather + all_gather + rms_norm_post_all_gather and will
// later be replaced with proper fused pre/post all-gather ops.
class DistributedRMSNormDecompositionRewritePattern
    : public OpRewritePattern<ttnn::DistributedRMSNormOp> {
public:
  using OpRewritePattern<ttnn::DistributedRMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::DistributedRMSNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMDECOMPOSITIONREWRITEPATTERN_H
