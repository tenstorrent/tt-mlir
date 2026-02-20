// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATRESHAPEELTWISESLICEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATRESHAPEELTWISESLICEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// When a RepeatOp feeds through a ReshapeOp and an element-wise binary op
// before being sliced, the full repeated tensor and the eltwise intermediate
// are materialized—often causing DRAM OOM.
//
// Pattern:  repeat -> reshape -> eltwise(other, %) -> [slice -> reshape]*
//
// If the slice operates on a dimension that maps 1:1 through the reshape
// and was NOT expanded by the repeat, the slice can be hoisted before
// the repeat.  The eltwise's other input is sliced on the same dimension.
//
// Example:
//   repeat(1x32x1x6x5120, 1x1x1560x1x1) -> 1x32x1560x6x5120
//   reshape -> 1x49920x6x5120
//   add(1x1x6x5120, 1x49920x6x5120) -> 1x49920x6x5120
//   slice(dim2:[i:i+1]) -> 1x49920x1x5120
//   reshape -> 49920x5120
// becomes (for each slice):
//   slice(1x32x1x6x5120, dim3:[i:i+1]) -> 1x32x1x1x5120
//   repeat(1x1x1560x1x1) -> 1x32x1560x1x5120
//   reshape -> 1x49920x1x5120
//   slice(1x1x6x5120, dim2:[i:i+1]) -> 1x1x1x5120
//   add -> 1x49920x1x5120
//   reshape -> 49920x5120
//
// This reduces peak memory by the factor of the sliced dimension (6x here).
class RepeatReshapeEltwiseSliceRewritePattern
    : public OpRewritePattern<ttnn::RepeatOp> {
public:
  using OpRewritePattern<ttnn::RepeatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::RepeatOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_REPEATRESHAPEELTWISESLICEREWRITEPATTERN_H
