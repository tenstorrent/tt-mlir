// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ALLGATHERMATMULFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ALLGATHERMATMULFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses matmul/linear(all_gather(input), weight) into a
// ttcore.composite "all_gather_minimal_matmul_async" (resolved later by
// TTNNResolveComposites). Templated on MatmulOp/LinearOp; the linear variant's
// bias rides along into the composite (matmul has no bias). Bails on
// transpose_a and multi-use gathers; transpose_b is materialized as a
// permute of the weight to [K, N]. Equal last-dim slices of the projection
// (Wan fused QKV) become `chunks=C` results. Defers to
// AllGatherMatmulAddcmulFusing when a gated-residual epilogue follows.
template <typename MatmulLikeOp>
class AllGatherMatmulFusing : public mlir::OpRewritePattern<MatmulLikeOp> {
public:
  using mlir::OpRewritePattern<MatmulLikeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatmulLikeOp matmulOp,
                  mlir::PatternRewriter &rewriter) const final;
};

// Fuses `residual + gate * matmul/linear(all_gather(input), weight)` into the
// same composite, mapping residual/gate to the addcmul operands (residual ->
// addcmul_input1, gate -> addcmul_input2) with scalar = 1.0. A leading-unit
// reshape between the projection and the multiply is skipped; captured
// residual/gate are squeezed to the 2D projection rank.
// Anchored on AddOp; templated on the projection op (MatmulOp/LinearOp).
template <typename MatmulLikeOp>
class AllGatherMatmulAddcmulFusing : public mlir::OpRewritePattern<AddOp> {
public:
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ALLGATHERMATMULFUSINGPATTERN_H
