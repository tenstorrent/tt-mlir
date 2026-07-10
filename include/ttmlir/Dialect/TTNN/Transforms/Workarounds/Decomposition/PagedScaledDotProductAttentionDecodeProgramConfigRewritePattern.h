// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// On Blackhole, sets exp_approx_mode = false on the op's SDPAProgramConfig. The
// tt-metal default approx-exp path fails SFPI compile on Blackhole
// (tt-metal #40301). tt-metal already fills every other field with a sensible
// default when no config is passed (q/k_chunk_size = 32,
// max_cores_per_head_batch = 1), so no config is needed off Blackhole; but
// because passing any config disables metal's auto-default, on Blackhole we
// replicate those defaults for the remaining fields (or preserve an existing
// config) while forcing exp_approx_mode = false.
//
// The exp_approx_mode override is an invariant: on Blackhole it is applied to
// whatever program_config the op ends up with -- IR-provided, optimizer-set, or
// synthesized here -- so the pattern re-fires to fixpoint if another pattern
// rebuilds the op with a fresh config.
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
