// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttir::fusing {

// Fuses Sort + Slice into a single TopK operation at the TTIR level.
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
// This pattern is unconditional — no op-model validation is performed.
// Validation happens later at the TTNN level via
// TopKDecompositionRewritePattern.
class TopKFusingPattern : public mlir::OpRewritePattern<SortOp> {
public:
  using OpRewritePattern<SortOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SortOp srcOp, mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_TOPKFUSINGPATTERN_H
