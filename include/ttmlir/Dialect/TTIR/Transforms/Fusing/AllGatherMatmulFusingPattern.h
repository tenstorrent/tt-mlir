// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ALLGATHERMATMULFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ALLGATHERMATMULFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses an all_gather feeding a matmul/linear into a
//   ttcore.composite "all_gather_minimal_matmul_async"
// whose decomposition body reproduces the primitive form:
//
//   gathered = all_gather(x)
//   proj     = matmul(gathered, W)                 (or linear(gathered, W, b))
//
// Anchored on the matmul/linear (templated on MatmulOp or LinearOp). Defers to
// AllGatherMatmulAddcmulFusing when a gated-residual epilogue follows, so the
// whole epilogue folds into one composite. Bails on transposed operands the
// fused op cannot model, and requires the gathered value to have a single use
// so the collective is not duplicated.
//
// Promotion to ttnn.all_gather_minimal_matmul_async (or inlining the
// decomposition body as a fallback) happens later at the TTNN level via
// TTNNResolveComposites. Matching at the TTIR level keeps the graph free of the
// metal-restricted layout/memory ops that clutter the TTNN level.
template <typename MatmulLikeOp>
class AllGatherMatmulFusing : public mlir::OpRewritePattern<MatmulLikeOp> {
public:
  using mlir::OpRewritePattern<MatmulLikeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatmulLikeOp matmulOp,
                  mlir::PatternRewriter &rewriter) const final;
};

// Fuses an all_gather + matmul/linear + gated-residual epilogue
//   out = residual + gate * matmul(all_gather(x), W)
// into ttcore.composite "all_gather_minimal_matmul_async" with residual/gate
// mapped to the addcmul operands and scalar = 1.0. Anchored on AddOp; templated
// on the projection op (MatmulOp or LinearOp). Requires single uses of the
// gather, projection, and multiply so nothing is duplicated.
template <typename MatmulLikeOp>
class AllGatherMatmulAddcmulFusing : public mlir::OpRewritePattern<AddOp> {
public:
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOp addOp, mlir::PatternRewriter &rewriter) const final;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_ALLGATHERMATMULFUSINGPATTERN_H
