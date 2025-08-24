// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATENATEHEADSOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATENATEHEADSOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ConcatenateHeadsOp rewrite pattern that rewrites the operation
// into equivalent ttnn::PermuteOp + ttnn::ReshapeOp sequence.
//
// Input tensor shape: [batch_size, num_heads, sequence_size, head_size]
// Output tensor shape: [batch_size, sequence_size, num_heads * head_size]
//
// Rewrite strategy:
// 1. Permute: [batch_size, num_heads, sequence_size, head_size]
//             -> [batch_size, sequence_size, num_heads, head_size]
// 2. Reshape: [batch_size, sequence_size, num_heads, head_size]
//             -> [batch_size, sequence_size, num_heads * head_size]
class ConcatenateHeadsOpRewritePattern
    : public OpRewritePattern<ttnn::ConcatenateHeadsOp> {
public:
  using OpRewritePattern<ttnn::ConcatenateHeadsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ConcatenateHeadsOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONCATENATEHEADSOPREWRITEPATTERN_H
