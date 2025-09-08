// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir::utils {
llvm::SmallVector<int64_t> unsqueezeValue(mlir::PatternRewriter &rewriter,
                                          mlir::Location loc,
                                          mlir::Value &input,
                                          mlir::RankedTensorType desiredType,
                                          bool frontUnsqueeze) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> unsqueezeShape(desiredType.getRank(), 1);
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    int64_t idx =
        frontUnsqueeze ? (desiredType.getRank() - inputType.getRank()) + i : i;
    unsqueezeShape[idx] = inputType.getDimSize(i);
  }

  llvm::SmallVector<int32_t> reshapeDim(unsqueezeShape.begin(),
                                        unsqueezeShape.end());

  auto reshapeDimAttr = rewriter.getI32ArrayAttr(reshapeDim);
  input = createDPSOp<ttir::ReshapeOp>(
      rewriter, loc, unsqueezeShape, desiredType.getElementType(),
      desiredType.getEncoding(), input, reshapeDimAttr);
  return unsqueezeShape;
}

// Calculate a reblocking affine map from inputShape to outputShape.
mlir::AffineMap calculateReblockMap(mlir::ArrayRef<int64_t> inputShape,
                                    mlir::ArrayRef<int64_t> outputShape,
                                    mlir::MLIRContext *ctx) {
  assert(inputShape.size() == outputShape.size() && "Rank must be preserved");

  size_t rank = inputShape.size();
  assert(rank % 2 == 0);
  size_t halfRank = rank / 2;

  mlir::ArrayRef<int64_t> inputShardShape = inputShape.drop_front(halfRank);
  mlir::ArrayRef<int64_t> outputGridShape = outputShape.take_front(halfRank);
  mlir::ArrayRef<int64_t> outputShardShape = outputShape.drop_front(halfRank);

  mlir::SmallVector<mlir::AffineExpr> mapExprs(rank);

  for (size_t i = 0; i < halfRank; i++) {
    auto dG = getAffineDimExpr(i, ctx);
    mapExprs[i] = dG.floorDiv(outputGridShape[i]);

    size_t j = i + halfRank;
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[j] = dG * outputShardShape[i] + dS;
  }
  auto outputToCanonical = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  for (size_t i = 0; i < halfRank; i++) {
    size_t j = i + halfRank;
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[i] = dS.floorDiv(inputShardShape[i]);
    mapExprs[j] = dS % inputShardShape[i];
  }
  auto canonicalToInput = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  return canonicalToInput.compose(outputToCanonical);
}

mlir::LogicalResult broadcastValue(mlir::PatternRewriter &rewriter,
                                   mlir::Value input,
                                   mlir::RankedTensorType desiredType,
                                   mlir::Value &output, mlir::Location loc,
                                   bool frontUnsqueeze) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> inputShape(inputType.getShape());
  llvm::SmallVector<int64_t, 4> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(
          inputShape, desiredType.getShape(), broadcastedShape)) {
    return mlir::failure();
  }

  if (inputShape == desiredType.getShape()) {
    output = input;
    return mlir::success();
  }

  if (inputType.getRank() != desiredType.getRank()) {
    inputShape =
        unsqueezeValue(rewriter, loc, input, desiredType, frontUnsqueeze);
  }

  llvm::SmallVector<int64_t> broadcastDims =
      ttmlir::utils::getBroadcastDimensions<int64_t>(inputShape,
                                                     desiredType.getShape());

  output = createDPSOp<ttir::BroadcastOp>(rewriter, loc, desiredType, input,
                                          broadcastDims);
  return mlir::success();
}

llvm::SmallVector<int64_t, 2>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  const int64_t minGridValue = *llvm::min_element(targetGridShape);

  llvm::SmallVector<int64_t, 2> squareGrid(targetGridShape.size(),
                                           minGridValue);
  return squareGrid;
}

} // namespace mlir::tt::ttir::utils
