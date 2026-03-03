// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal's ttnn::pad only supports padding on the lowest 3 dimensions for
// rank > 4 tensors, and has a bug (update_original_shape) that silently
// corrupts shapes when unsqueezing padded tensors back to rank > 4.
//
// This pattern handles any rank > 4 PadOp with at most 3 padded dimensions
// by squeezing to a lower rank:
//   1. Permute non-padded dims to front, padded dims to back
//   2. Reshape to merge all non-padded dims into one (rank <= 4)
//   3. Pad the reduced-rank tensor
//   4. Reshape back to original rank
//   5. Inverse permute to restore original order
// Issue: https://github.com/tenstorrent/tt-metal/issues/38144
class PadHighDimRewritePattern : public mlir::OpRewritePattern<PadOp> {
public:
  using mlir::OpRewritePattern<PadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(PadOp srcOp,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H
