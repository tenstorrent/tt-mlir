// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_POINTTOPOINTOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_POINTTOPOINTOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ttnn::PointToPointOp does not support transporting tensors across 2D mesh
// coordinates. For example, ttnn.point_to_point(tensor, [0, 0], [1, 1]) is not
// supported. It only allows to transport across MeshCoordinate where Row or
// Columm are the same. For example, ttnn.point_to_point(tensor, [0, 0], [0, 1])
// is supported because Row is the same. This rewrite pattern decomposes the
// PointToPointOp into a sequence of PointToPointOps where the Row or Column are
// the same. For example, ttnn.point_to_point(tensor, [0, 0], [1, 1]) will be
// decomposed into: ttnn.point_to_point(tensor, [0, 0], [0, 1])
// ttnn.point_to_point(tensor, [0, 1], [1, 1])
// This rewrite pattern is only supported for 2D mesh coordinates.
// See https://github.com/tenstorrent/tt-metal/issues/33924 for the issue.
class PointToPointOpRewritePattern
    : public OpRewritePattern<ttnn::PointToPointOp> {
public:
  using OpRewritePattern<ttnn::PointToPointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::PointToPointOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_POINTTOPOINTOPREWRITEPATTERN_H
