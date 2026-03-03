// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Fuses the RoPE (Rotary Position Embedding) subgraph into a single
// RotaryEmbeddingOp.
//
// Matches:  (x * cos) + (rotate_half(x) * sin)
// Produces: rotary_embedding(x, cos, sin)
class RoPEFusing : public mlir::OpRewritePattern<AddOp> {
public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const override;
};

// Commute a downstream permute through an already-fused RotaryEmbeddingOp
// by switching to decode mode (token_index=0).
//
// Matches:  rotary_embedding(x, cos, sin) -> permute {2, 0, 1, 3}
// Produces: permute(x, {2, 0, 1, 3}) -> rotary_embedding(..., token_index=0)
class RoPEDecodeFusing : public mlir::OpRewritePattern<PermuteOp> {
public:
  using OpRewritePattern<PermuteOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(PermuteOp permuteOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H
