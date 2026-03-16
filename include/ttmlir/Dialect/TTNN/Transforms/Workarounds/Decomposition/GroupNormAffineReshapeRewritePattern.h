// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GROUPNORMAFFINERESHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GROUPNORMAFFINERESHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Reshape GroupNorm affine operands from 1D (C,) to 4D (1, 1, C/32, 32) so
// all backend targets use a consistent representation.
class GroupNormAffineReshapeRewritePattern
    : public OpRewritePattern<ttnn::GroupNormOp> {
public:
  using OpRewritePattern<ttnn::GroupNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::GroupNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GROUPNORMAFFINERESHAPEREWRITEPATTERN_H
