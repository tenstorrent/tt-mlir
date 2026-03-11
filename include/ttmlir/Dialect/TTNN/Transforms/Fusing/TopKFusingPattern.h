// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Fuses the Sort + Slice pattern into a single TopKOp.
//
// The pattern anchors on SortOp and looks for SliceStaticOp users on one or
// both of its results (values and/or indices). At least one result must be
// sliced; the other may be unused. When both are sliced, their slice
// parameters must be identical.
//
// Matches the following slice positions:
//   Slice from start: slice([0..k]) on the sorted dimension
//   Slice from end:   slice([n-k..n]) on the sorted dimension
//
// For slice-from-end, the `largest` flag is inverted relative to the sort's
// `descending` flag, since taking the tail of a descending sort yields
// the smallest elements (and vice versa).
//
// Produces: topk(input, k) -> (top_values, top_indices)
// Unused TopK results become dead values.
class TopKFusing : public mlir::OpRewritePattern<SortOp> {
public:
  TopKFusing(mlir::MLIRContext *context,
             const FusionValidationConfig &validationConfig = {})
      : OpRewritePattern<SortOp>(context), validationConfig(validationConfig) {}

  mlir::LogicalResult
  matchAndRewrite(SortOp srcOp, mlir::PatternRewriter &rewriter) const override;

private:
  FusionValidationConfig validationConfig;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H
