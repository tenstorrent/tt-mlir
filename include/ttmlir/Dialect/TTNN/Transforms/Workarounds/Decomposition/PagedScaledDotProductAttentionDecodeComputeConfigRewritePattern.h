// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODECONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODECONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which adds a high-precision DeviceComputeKernelConfig to
// PagedScaledDotProductAttentionDecodeOp when no config is present.
// This sets specific math fidelity and fp32_dest_acc_en=true for improved
// numerical accuracy. This addresses potential numerical precision issues in
// paged SDPA decode operations, similar to the workaround applied to
// ReduceScatterOp.
class PagedScaledDotProductAttentionDecodeComputeConfigRewritePattern
    : public OpRewritePattern<ttnn::PagedScaledDotProductAttentionDecodeOp> {
public:
  using OpRewritePattern<
      ttnn::PagedScaledDotProductAttentionDecodeOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(ttnn::PagedScaledDotProductAttentionDecodeOp srcOp,
                  PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODECONFIGREWRITEPATTERN_H
