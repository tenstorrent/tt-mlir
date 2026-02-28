// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PointToPointOpRewritePattern.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ttnn::PointToPointOp does not support transporting tensors across 2D mesh
// coordinates. For example, ttnn.point_to_point(tensor, [0, 0], [1, 1]) is not
// supported. It only allows to transport across MeshCoordinate where Row or
// Column are the same. For example, ttnn.point_to_point(tensor, [0, 0], [0, 1])
// is supported because Row is the same. This rewrite pattern decomposes the
// PointToPointOp into a sequence of PointToPointOps where the Row or Column are
// the same. For example, ttnn.point_to_point(tensor, [0, 0], [1, 1]) will be
// decomposed into: ttnn.point_to_point(tensor, [0, 0], [0, 1])
// ttnn.point_to_point(tensor, [0, 1], [1, 1])
// This rewrite pattern is only supported for 2D mesh coordinates.
// See https://github.com/tenstorrent/tt-metal/issues/33924 for the issue.

LogicalResult
PointToPointOpRewritePattern::matchAndRewrite(ttnn::PointToPointOp srcOp,
                                              PatternRewriter &rewriter) const {
  const auto &senderCoord = srcOp.getSenderCoord();
  const auto &receiverCoord = srcOp.getReceiverCoord();
  if ((senderCoord[0] == receiverCoord[0]) ||
      (senderCoord[1] == receiverCoord[1])) {
    // If Row or Column are the same, no decomposition is needed.
    return failure();
  }

  // Decompose the PointToPointOp into a sequence of PointToPointOps
  ::llvm::SmallVector<int64_t, 2> intermediateCoordVec{senderCoord[0],
                                                       receiverCoord[1]};
  ::llvm::ArrayRef<int64_t> intermediateCoord = intermediateCoordVec;

  auto p2pOp1 = rewriter.create<ttnn::PointToPointOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                          "_p2p_to_intermediate"),
      srcOp.getResult().getType(), srcOp.getInput(), senderCoord,
      intermediateCoord, /*optional_output_tensor=*/nullptr);
  auto optionalOutputTensor = srcOp.getOptionalOutputTensor();
  auto p2pOp2 = rewriter.create<ttnn::PointToPointOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                          "_p2p_from_intermediate"),
      srcOp.getResult().getType(), p2pOp1.getResult(), intermediateCoord,
      receiverCoord, optionalOutputTensor ? optionalOutputTensor : nullptr);
  rewriter.replaceOp(srcOp, p2pOp2.getResult());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
