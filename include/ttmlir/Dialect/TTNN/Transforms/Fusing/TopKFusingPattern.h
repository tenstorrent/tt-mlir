// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::fusing {

// Fuses the Sort + Slice + Slice pattern into a single TopKOp.
//
// Matches:  sort(input) -> (values, indices)
//           slice(values, [0..k]) -> top_values
//           slice(indices, [0..k]) -> top_indices
//
// Produces: topk(input, k) -> (top_values, top_indices)
//
// The pattern anchors on SortOp and looks for two SliceStaticOp users
// (one for values, one for indices) with identical slice parameters.
class TopKFusing : public mlir::OpRewritePattern<SortOp> {
public:
  using OpRewritePattern<SortOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SortOp srcOp, mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H
