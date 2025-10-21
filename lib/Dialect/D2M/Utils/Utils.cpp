// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Utils.h"
#include "ttmlir/Asserts.h"

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

Value getPhysicalTensorOrMemref(mlir::Value tensorOrMemref) {

  TT_assertv((mlir::isa<mlir::RankedTensorType>(tensorOrMemref.getType()) ||
              mlir::isa<mlir::MemRefType>(tensorOrMemref.getType())),
             "Expected a tensor or memref type");
  auto physTensor = tensorOrMemref;

  if (auto viewOp = mlir::dyn_cast<mlir::tt::d2m::ViewOpInterface>(
          tensorOrMemref.getDefiningOp())) {
    physTensor = viewOp.getInput();
  } else if (auto toLayoutOp = mlir::dyn_cast<mlir::tt::d2m::ToLayoutOp>(
                 tensorOrMemref.getDefiningOp())) {
    physTensor = toLayoutOp.getInitOperand();

    if (auto viewOp = mlir::dyn_cast<mlir::tt::d2m::ViewOpInterface>(
            physTensor.getDefiningOp())) {
      physTensor = viewOp.getInput();
    }
  } else if (auto genericOp = mlir::dyn_cast<mlir::tt::d2m::GenericOp>(tensorOrMemref.getDefiningOp())) {
    // Assume that if the defining op is a generic op, the output is the first of the outputs.
    auto genericOutput = genericOp.getOutputs()[0];
    physTensor = getPhysicalTensorOrMemref(genericOutput);
  }

  return physTensor;
}

ttcore::DeviceLayoutInterface
getDeviceLayoutInterfaceIfExists(mlir::Value tensorOrMemref) {
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(tensorOrMemref.getType())) {
    return mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
        tensorType.getEncoding());
  } else if (auto memrefType =
                 mlir::dyn_cast<mlir::MemRefType>(tensorOrMemref.getType())) {
    return mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
        memrefType.getLayout());
  } else {
    return nullptr;
  }
}

SmallVector<int64_t> getGridShape(mlir::Value tensorOrMemref) {
  auto shapedType = mlir::dyn_cast<ShapedType>(tensorOrMemref.getType());
  TT_assertv(shapedType, "Expected a shaped type");

  // Assume a default grid shape of {1, 1} if the layout is not found.
  auto layout = getDeviceLayoutInterfaceIfExists(tensorOrMemref);
  return (layout) ? llvm::SmallVector<int64_t>(layout.getGridShape(shapedType))
                  : llvm::SmallVector<int64_t>({1, 1});
}

SmallVector<int64_t> getPhysicalGridShape(mlir::Value tensorOrMemref) {
  auto physTensorOrMemref = getPhysicalTensorOrMemref(tensorOrMemref);
  auto gridShape = getGridShape(physTensorOrMemref);
  return gridShape;
}

} // namespace mlir::tt::d2m::utils
