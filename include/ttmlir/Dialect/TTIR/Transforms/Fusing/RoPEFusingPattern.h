// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses: (x * cos) + (rotate_half(x) * sin)
//        -> ttcore.composite "rotary_embedding" (x, cos, sin)
//   where rotate_half(x) = concat(neg(x[D/2:]), x[:D/2])
//
// Anchors on AddOp. Handles commuted operand orders for both add and multiply.
// Traces cos/sin through TM chains (TypecastOp, ReshapeOp, BroadcastOp).
//
// This pattern is unconditional — no op-model validation is performed.
// Validation happens later at the TTNN level via TTNNResolveComposites.
class RoPERotateHalfFusingPattern : public mlir::OpRewritePattern<AddOp> {
public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const override;
};

// Fuses: concat(sub(x1*cos, x2*sin), add(x2*cos, x1*sin))
//        -> ttcore.composite "rotary_embedding"
//             (x, concat(cos_h, cos_h), concat(sin_h, sin_h))
//   where x1 = x[:D/2], x2 = x[D/2:]
//
// Anchors on ConcatOp. Handles optional pre-scaling of cos/sin (the scaled
// values are absorbed as the cache inputs).
//
// This pattern is unconditional — no op-model validation is performed.
class RoPEComplexRotationFusingPattern
    : public mlir::OpRewritePattern<ConcatOp> {
public:
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConcatOp srcOp,
                  mlir::PatternRewriter &rewriter) const override;
};

// Fuses interleaved-pair RoPE:
//   x_   = reshape(x, [..., D/2, 1, 2])
//   out  = reshape(freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1],
//                  [..., D])
//   where freqs is shape (..., D/2, 2, 2) packing per-pair
//   [[cos,-sin],[sin,cos]]
//
// Rewrites to rotate-half form so it lowers to the existing rotary_embedding
// op:
//   x_rh   = cat([x[..., 0::2], x[..., 1::2]], dim=-1)
//   cos    = cat([freqs[..., 0, 0], freqs[..., 0, 0]], dim=-1)
//   sin    = cat([freqs[..., 1, 0], freqs[..., 1, 0]], dim=-1)
//   out_rh = rotary_embedding(x_rh, cos, sin)
//   result = reshape(permute(reshape(out_rh, [..., 2, D/2]), [0,1,2,4,3]),
//                    [..., D])
//
// Anchors on AddOp. Walks back through Broadcast/Reshape to identify the
// 6D reshape of x and the freqs_cis slices.
class RoPEInterleavedPairFusingPattern : public mlir::OpRewritePattern<AddOp> {
public:
  using OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ROPEFUSINGPATTERN_H
