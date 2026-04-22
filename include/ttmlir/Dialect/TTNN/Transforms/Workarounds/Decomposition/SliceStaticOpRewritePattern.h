// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SLICESTATICOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SLICESTATICOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// tt-metal's slice op fails when the circular buffer for the last output
// dimension exceeds L1 memory (check: output_shape[-1] * elem_size * 2 >
// l1_size). This pattern decomposes such a slice into permute -> slice ->
// permute, swapping the last dimension with a smaller-output dimension so
// the CB fits in L1.
class SliceStaticOpRewritePattern
    : public OpRewritePattern<ttnn::SliceStaticOp> {
public:
  using OpRewritePattern<ttnn::SliceStaticOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SliceStaticOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SLICESTATICOPREWRITEPATTERN_H
