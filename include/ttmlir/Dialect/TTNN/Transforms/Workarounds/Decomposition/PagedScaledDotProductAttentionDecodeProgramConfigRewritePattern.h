// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround that sets an explicit SDPAProgramConfig on
// PagedScaledDotProductAttentionDecodeOp when head_dim is large enough that
// the default TTNN schedule overflows per-core L1 (Gemma-4 head_dim=512
// global attention and head_dim=256 sliding-window layers). Pins
// k_chunk_size=32 (one page) and caps max_cores_per_head_batch at 32 to
// halve the partial-accumulator CB footprint vs the 64-core tree-reduction
// limit. Other configurations are left unset so the runtime / TTNN default
// schedule applies.
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
