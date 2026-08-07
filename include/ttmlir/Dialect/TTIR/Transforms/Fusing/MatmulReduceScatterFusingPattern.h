// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_MATMULREDUCESCATTERFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_MATMULREDUCESCATTERFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses reduce_scatter(matmul/linear(input, weight)) into a
// ttcore.composite "minimal_matmul_strided_reduce_scatter_async" (resolved
// later by TTNNResolveComposites). Templated on MatmulOp/LinearOp; the linear
// variant's bias rides along into the composite (matmul has no bias). Anchored
// on ReduceScatterOp. Bails on transposed operands and multi-use projections;
// defers to MatmulReduceScatterAddcmulFusing when a gated-residual epilogue
// follows.
template <typename MatmulLikeOp>
class MatmulReduceScatterFusing
    : public mlir::OpRewritePattern<ReduceScatterOp> {
public:
  using mlir::OpRewritePattern<ReduceScatterOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReduceScatterOp reduceScatterOp,
                  mlir::PatternRewriter &rewriter) const final;
};

// Fuses `residual + gate * reduce_scatter(matmul/linear(input, weight))` into
// the same composite, mapping residual/gate to the addcmul operands (residual
// -> addcmul_input1, gate -> addcmul_input2) with scalar = 1.0. Anchored on
// AddOp; templated on the projection op (MatmulOp/LinearOp).
template <typename MatmulLikeOp>
class MatmulReduceScatterAddcmulFusing : public mlir::OpRewritePattern<AddOp> {
public:
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_MATMULREDUCESCATTERFUSINGPATTERN_H
