// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMRESHAPETOCANONICALSHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMRESHAPETOCANONICALSHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Workaround that reshapes the input/residual of a DistributedRMSNormOp
// into the canonical (1, 1, 32, M) shape required by the tt-metal
// fused_rms_minimal kernel and reshapes the result back to the original
// shape afterwards.
//
// Matches when the op is otherwise eligible for the fused kernel (dim -2
// equals 32, dim -1 is a multiple of 32, all leading dims are 1, weight is
// present) but the input rank is not 4 or its leading dims are not in the
// (1, 1, 32, M) canonical layout. Must run before
// DistributedRMSNormWidthShardInputRewritePattern so the latter sees the
// canonical 4D shape; this is enforced via a higher pattern benefit.
class DistributedRMSNormReshapeToCanonicalShapeRewritePattern
    : public OpRewritePattern<ttnn::DistributedRMSNormOp> {
public:
  using OpRewritePattern<ttnn::DistributedRMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::DistributedRMSNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_DISTRIBUTEDRMSNORMRESHAPETOCANONICALSHAPEREWRITEPATTERN_H
