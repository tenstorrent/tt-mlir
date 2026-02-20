// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATSLICERESHAPEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATSLICERESHAPEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// When a RepeatOp expands dimensions that are NOT in the last two positions
// (the tile-aligned dims), the resulting tensor can have non-tile-aligned
// inner dimensions, causing DRAM OOM during allocation.
//
// This pattern captures: repeat -> slice_static -> reshape
// and rewrites it to: slice_static -> repeat -> reshape
// by hoisting the slice before the repeat, so the repeat operates on a
// smaller tensor with tile-friendly dimensions.
//
// Example:
//   repeat(1x1x6x5120 -> 1x49920x6x5120) -> slice(dim2:[1:2]) -> reshape
// becomes:
//   slice(1x1x6x5120, dim2:[1:2]) -> 1x1x1x5120
//   repeat(1x1x1x5120 -> 1x49920x1x5120)
//   reshape(1x49920x1x5120 -> 49920x5120)
//
// The pattern backs off if the final reshape output shape is inconsistent
// with the rewritten sequence.
class RepeatSliceReshapeRewritePattern
    : public OpRewritePattern<ttnn::RepeatOp> {
public:
  using OpRewritePattern<ttnn::RepeatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RepeatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATSLICERESHAPEREWRITEPATTERN_H
