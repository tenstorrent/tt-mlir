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

// Fuses the Sort + Slice + Slice pattern into a single TopKOp.
//
// Matches two cases:
//   Case 1 (slice from start):
//     sort(input) -> (values, indices)
//     slice(values, [0..k]) -> top_values
//     slice(indices, [0..k]) -> top_indices
//   Case 2 (slice from end):
//     sort(input) -> (values, indices)
//     slice(values, [n-k..n]) -> top_values
//     slice(indices, [n-k..n]) -> top_indices
//
// Produces: topk(input, k) -> (top_values, top_indices)
//
// For case 2, the `largest` flag is inverted relative to the sort's
// `descending` flag, since taking the tail of a descending sort yields
// the smallest elements (and vice versa).
//
// The pattern anchors on SortOp and looks for two SliceStaticOp users
// (one for values, one for indices) with identical slice parameters.
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
