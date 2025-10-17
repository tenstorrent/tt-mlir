// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m::utils {

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

  auto map = canonicalToInput.compose(outputToCanonical);
  return map;
}

llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  const int64_t minGridValue = *llvm::min_element(targetGridShape);

  llvm::SmallVector<int64_t, 2> squareGrid(targetGridShape.size(),
                                           minGridValue);
  return squareGrid;
}

Value getPhysicalTensor(mlir::Value tensor) {

  auto physTensor = tensor;
  if (auto viewOp = mlir::dyn_cast<mlir::tt::d2m::ViewOpInterface>(
          tensor.getDefiningOp())) {
    physTensor = viewOp.getInput();
  } else if (auto toLayoutOp = mlir::dyn_cast<mlir::tt::d2m::ToLayoutOp>(
                 tensor.getDefiningOp())) {
    physTensor = toLayoutOp.getInitOperand();
    if (auto viewOp = mlir::dyn_cast<mlir::tt::d2m::ViewOpInterface>(
            physTensor.getDefiningOp())) {
      physTensor = viewOp.getInput();
    }
  }
  return physTensor;
}

llvm::SmallVector<int64_t> getPhysicalGridShape(mlir::Value tensor) {

  auto physTensor = getPhysicalTensor(tensor);

  llvm::SmallVector<int64_t> result;
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(physTensor.getType())) {
    auto layout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
        tensorType.getEncoding());
    result = llvm::SmallVector<int64_t>(layout.getGridShape(tensorType));
  } else if (auto memrefType =
                 mlir::dyn_cast<mlir::MemRefType>(physTensor.getType())) {
    auto layout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
        memrefType.getLayout());
    result = llvm::SmallVector<int64_t>(layout.getGridShape(memrefType));
  } else {
    result = {1, 1};
  }

  return result;
}

} // namespace mlir::tt::d2m::utils
