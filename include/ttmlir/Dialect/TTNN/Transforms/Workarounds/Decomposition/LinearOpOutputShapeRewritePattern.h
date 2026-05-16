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
// TILE_HEIGHT), tt-metal may return the matmul rank or the bias-broadcast rank
// depending on how the op was typed. This pattern inserts an explicit
// linear + reshape so downstream consumers (e.g. ttnn.concat) always see the
// rank declared in the IR:
//   - broadcast-typed linear: linear(matmul rank) -> reshape(broadcast rank)
//   - matmul-typed linear:    linear(broadcast rank) -> reshape(matmul rank)
// See: https://github.com/tenstorrent/tt-metal/issues/39392
//      https://github.com/tenstorrent/tt-xla/issues/4633
class LinearOpOutputShapeRewritePattern
    : public OpRewritePattern<ttnn::LinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::LinearOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_LINEAROPOUTPUTSHAPEREWRITEPATTERN_H
