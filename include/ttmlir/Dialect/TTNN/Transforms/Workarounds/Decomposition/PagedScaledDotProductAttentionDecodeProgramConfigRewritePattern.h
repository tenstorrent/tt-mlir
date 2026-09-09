// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround which, on Blackhole, forces exp_approx_mode = false on the SDPA
// decode program config; the tt-metal default approx-exp path fails SFPI
// compile. Metal issue reference:
// https://github.com/tenstorrent/tt-metal/issues/40301
// An existing config is preserved, otherwise metal's defaults are replicated
// (q/k_chunk_size = 32, max_cores_per_head_batch = 1), since passing a config
// disables metal's auto-default.
class PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern
    : public OpRewritePattern<ttnn::PagedScaledDotProductAttentionDecodeOp> {
public:
  using OpRewritePattern<
      ttnn::PagedScaledDotProductAttentionDecodeOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(ttnn::PagedScaledDotProductAttentionDecodeOp srcOp,
                  PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H
