// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SPLITQUERYKEYVALUEANDSPLITHEAD\
SOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SPLITQUERYKEYVALUEANDSPLITHEAD\
SOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// SplitQueryKeyValueAndSplitHeadsOp rewrite pattern that rewrites the operation
// into its constituent operations when head_size is not divisible by tile size.
//
// This pattern undoes the fusion performed by
// SplitQueryKeyValueAndSplitHeadsUpdatePattern in TTIRFusing.cpp when the
// head_size dimension is not divisible by the tile size (32).
//
// The SplitQueryKeyValueAndSplitHeadsOp has a limitation: it requires head_size
// to be divisible by the tile size (32). When this constraint is not met, this
// pattern decomposes the fused operation back into its individual operations.
//
// Input tensor shape: [batch_size, sequence_size, 3 * hidden_size]
// Output tensor shapes:
//   - query: [batch_size, num_heads, sequence_size, head_size]
//   - key: [batch_size, num_heads, sequence_size, head_size] or
//          [batch_size, num_heads, head_size, sequence_size] (if transpose_key
//          is true)
//   - value: [batch_size, num_heads, sequence_size, head_size]
//
// Rewrite strategy (reverses the fusion):
// 1. Split the input tensor into three separate tensors (Q, K, V) along the
// hidden dimension
// 2. For each of Q, K, V:
//    a. Reshape from [batch, seq, hidden] to [batch, seq, num_heads, head_size]
//    b. Permute to [batch, num_heads, seq, head_size]
// 3. If transpose_key is true, additionally permute K to [batch, num_heads,
// head_size, seq]
class SplitQueryKeyValueAndSplitHeadsOpRewritePattern
    : public OpRewritePattern<ttnn::SplitQueryKeyValueAndSplitHeadsOp> {
public:
  using OpRewritePattern<
      ttnn::SplitQueryKeyValueAndSplitHeadsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SplitQueryKeyValueAndSplitHeadsOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SPLITQUERYKEYVALUEANDSPLITHEADSOPREWRITEPATTERN_H
