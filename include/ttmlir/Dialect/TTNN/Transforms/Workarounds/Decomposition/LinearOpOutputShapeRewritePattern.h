// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LINEAROPOUTPUTSHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LINEAROPOUTPUTSHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// When the fused LinearOp kernel is used (padded bias second-to-last dim ==
// TILE_HEIGHT), the hardware output shape is the matmul shape, not the
// broadcasted shape. This pattern adjusts the LinearOp output from the
// broadcasted shape to the matmul shape and inserts a ReshapeOp to restore the
// original shape.
// See: https://github.com/tenstorrent/tt-metal/issues/39392
class LinearOpOutputShapeRewritePattern
    : public OpRewritePattern<ttnn::LinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::LinearOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LINEAROPOUTPUTSHAPEREWRITEPATTERN_H
