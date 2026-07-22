// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/OpValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Fuses the rotate_half RoPE (Rotary Position Embedding) subgraph into a
// single RotaryEmbeddingOp.
//
// Matches:  (x * cos) + (rotate_half(x) * sin)
//   where rotate_half(x) = concat(neg(x[half:]), x[:half])
// Produces: rotary_embedding(x, cos, sin)
class RoPERotateHalfFusing : public mlir::OpRewritePattern<AddOp> {
public:
  RoPERotateHalfFusing(mlir::MLIRContext *ctx, const OpValidationConfig &config)
      : OpRewritePattern<AddOp>(ctx), validationConfig(config) {}

  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const override;

private:
  OpValidationConfig validationConfig;
};

// Fuses the expanded (trig-identity) RoPE subgraph into a single
// RotaryEmbeddingOp.
//
// Matches:  concat(
//             subtract(x[:half] * cos_h, x[half:] * sin_h),
//             add(x[half:] * cos_h, x[:half] * sin_h))
//   where cos_h and sin_h are half-dim embeddings.
// Produces: rotary_embedding(x, concat(cos_h, cos_h), concat(sin_h, sin_h))
class RoPEExpandedFusing : public mlir::OpRewritePattern<ConcatOp> {
public:
  RoPEExpandedFusing(mlir::MLIRContext *ctx, const OpValidationConfig &config)
      : OpRewritePattern<ConcatOp>(ctx), validationConfig(config) {}

  mlir::LogicalResult
  matchAndRewrite(ConcatOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;

private:
  OpValidationConfig validationConfig;
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

// Same as RoPEDecodeFusing, but rooted on the canonicalized (folded) form of
// the decode permute: a layout-preserving reshape (BHSD with seq == 1 ->
// SBHD). A canonicalizer that runs before fusing folds the layout-preserving
// permute {2,0,1,3} into the adjacent reshape, so this sibling recognizes that
// reshape and applies the same decode rewrite.
//
// Matches:  rotary_embedding(x, cos, sin) -> reshape {[1,B,H,D]}
// Produces: permute(x, {2, 0, 1, 3}) -> rotary_embedding(..., token_index=0)
class RoPEDecodeReshapeFusing : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H
