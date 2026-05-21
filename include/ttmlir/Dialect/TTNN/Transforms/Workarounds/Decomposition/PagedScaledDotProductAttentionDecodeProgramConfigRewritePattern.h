// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Sets an explicit SDPAProgramConfig on PagedScaledDotProductAttentionDecodeOp
// for head_dim >= 256, where the default TTNN schedule overflows per-core L1.
// Scoped to opt-level=0; at opt-level>=1 OpValidationAndFallback owns this
// decision.
//
// Metal issue: https://github.com/tenstorrent/tt-metal/issues/44311
class PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern
    : public OpRewritePattern<ttnn::PagedScaledDotProductAttentionDecodeOp> {
public:
  PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern(
      MLIRContext *context, int64_t optimizationLevel)
      : OpRewritePattern<ttnn::PagedScaledDotProductAttentionDecodeOp>(context),
        optimizationLevel(optimizationLevel) {}

  LogicalResult
  matchAndRewrite(ttnn::PagedScaledDotProductAttentionDecodeOp srcOp,
                  PatternRewriter &rewriter) const override;

private:
  int64_t optimizationLevel;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PAGEDSCALEDDOTPRODUCTATTENTIONDECODEPROGRAMCONFIGREWRITEPATTERN_H
