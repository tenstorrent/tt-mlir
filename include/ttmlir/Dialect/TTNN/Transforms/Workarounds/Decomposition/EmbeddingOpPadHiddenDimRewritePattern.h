// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGOPPADHIDDENDIMREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGOPPADHIDDENDIMREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround for a ttnn.embedding miscompile.
//
// Empirical sweep on Blackhole (vocab=256, seq_len=64, bf16 weight,
// TILE_LAYOUT, DRAM) over ~160 tile_count values found:
//
//   raw ttnn.embedding PASSes iff
//       hidden_dim < 8192   OR   hidden_dim % 2048 == 0
//
// Anything else (e.g. Gemma-4 per-layer embedding D=10752, also seen at
// D=8320, 8704, 9216, 10752, 11008, ...) returns near-uncorrelated output
// (PCC ~ 0.02 vs torch.nn.functional.embedding).
//
// As a temporary workaround this pattern pads the weight's last dim up to
// the next multiple of 2048 (always in the known-good region for dims
// >= 8192), runs ttnn.embedding on the padded weight, then slices the
// output back to the original hidden dim. The proper fix belongs in
// tt-metal's ttnn::embedding kernel; this rewrite is intended to be
// removed once that fix lands.
class EmbeddingOpPadHiddenDimRewritePattern
    : public OpRewritePattern<ttnn::EmbeddingOp> {
public:
  using OpRewritePattern<ttnn::EmbeddingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::EmbeddingOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_EMBEDDINGOPPADHIDDENDIMREWRITEPATTERN_H
