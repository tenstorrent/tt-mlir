// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

namespace mlir::tt::d2m::utils {

namespace detail {

// Generates a permutation mask that interleaves groups the corresponding grid
SmallVector<int64_t> strideIterationOrder(int64_t rank) {
  TT_assertv(rank % 2 == 0, "Rank must be even");
  int64_t halfRank = rank / 2;
  SmallVector<int64_t> permutation;
  for (int64_t i = rank - 1; i >= 0; --i) {
    permutation.push_back((i % 2 * halfRank) + i / 2);
  }
  return permutation;
}
} // namespace detail

// Calculate a reblocking affine map from inputShape to outputShape.
mlir::AffineMap calculateReblockMap(mlir::ArrayRef<int64_t> inputShape,
                                    mlir::ArrayRef<int64_t> outputShape,
                                    mlir::MLIRContext *ctx) {

  int64_t inputRank = static_cast<int64_t>(inputShape.size());
  int64_t outputRank = static_cast<int64_t>(outputShape.size());
  TT_assertv(inputRank % 2 == 0, "Input rank must be even");
  TT_assertv(outputRank % 2 == 0, "Output rank must be even");

  if (inputShape == outputShape) {
    return mlir::AffineMap::getMultiDimIdentityMap(inputRank, ctx);
  }

  // Construct a map that transforms output (grid x shard) indices to logical
  // indices.
  mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, ctx);
  auto stride = mlir::getAffineConstantExpr(1, ctx);
  auto outputShapeIterationOrder = detail::strideIterationOrder(outputRank);
  for (const auto &i : outputShapeIterationOrder) {
    auto dim = mlir::getAffineDimExpr(i, ctx);
    expr = (dim * stride) + expr;
    stride = stride * outputShape[i];
  }
  auto outputToLogical = mlir::AffineMap::get(outputRank, 0, {expr}, ctx);

  // Construct a map that transforms logical indices to input (grid x shard)
  // indices.
  llvm::SmallVector<mlir::AffineExpr> toInputExprs(inputRank);
  stride = mlir::getAffineConstantExpr(1, ctx);
  auto dim = mlir::getAffineDimExpr(0, ctx);
  auto inputShapeIterationOrder = detail::strideIterationOrder(inputRank);
  for (const auto &i : inputShapeIterationOrder) {
    toInputExprs[i] = (dim.floorDiv(stride)) % inputShape[i];
    stride = stride * inputShape[i];
  }
  auto logicalToInput = mlir::AffineMap::get(1, 0, toInputExprs, ctx);

  return logicalToInput.compose(outputToLogical);
}

// Calculate a reblock affine map given a shape and new grid shape.
std::pair<mlir::SmallVector<int64_t>, mlir::AffineMap>
calculateReblockMapForGrid(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::ArrayRef<int64_t> newGridShape,
                           mlir::MLIRContext *context) {
  assert(tensorShape.size() % 2 == 0 &&
         "Expected even rank for grid + shard dimensions");
  assert(newGridShape.size() == tensorShape.size() / 2 &&
         "New grid shape must match grid rank of tensor shape");
  mlir::SmallVector<int64_t> newTensorShape(tensorShape);
  for (size_t i = 0; i < newGridShape.size(); i++) {
    size_t j = i + newGridShape.size();
    assert((tensorShape[i] * tensorShape[j]) % newGridShape[i] == 0 &&
           "New grid shape must evenly divide tensor shape");
    newTensorShape[j] = tensorShape[i] * tensorShape[j] / newGridShape[i];
    newTensorShape[i] = newGridShape[i];
  }
  return {newTensorShape,
          calculateReblockMap(tensorShape, newTensorShape, context)};
}

llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  const int64_t minGridValue = *llvm::min_element(targetGridShape);

  llvm::SmallVector<int64_t, 2> squareGrid(targetGridShape.size(),
                                           minGridValue);
  return squareGrid;
}

Type getRegionLargestDstElemType(Region &region) {
  auto getTypeNumberOfBits = [](Type type) {
    return ttcore::getNumberOfBits(ttcore::elementTypeToDataType(type));
  };

  Type largestType = nullptr;
  region.walk([&](OperandLoadStoreRegisterOpInterface op) {
    for (auto [operandIdx, v] :
         llvm::enumerate(op.getOperation()->getOperands())) {
      // Skip scalar operands.
      if (op.isScalarOperand(operandIdx)) {
        continue;
      }

      Type t = ttcore::getOperandInnerElementType(v);

      if (!largestType ||
          (getTypeNumberOfBits(t) > getTypeNumberOfBits(largestType))) {
        largestType = t;
      }

      if (largestType && getTypeNumberOfBits(largestType) >= 32u) {
        return WalkResult::interrupt();
      }
    }
    // Check output type for typecast operations that cast to a larger type.
    if (op.getOperation()->getNumResults() > 0) {
      Type outputType =
          ttcore::getOperandInnerElementType(op.getOperation()->getResult(0));
      if (!largestType || (getTypeNumberOfBits(outputType) >
                           getTypeNumberOfBits(largestType))) {
        largestType = outputType;
      }
      if (largestType && getTypeNumberOfBits(largestType) >= 32u) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  assert(largestType);
  TT_assert(getTypeNumberOfBits(largestType) <= 32u);
  return largestType;
}

RankedTensorType reblockTensor(RankedTensorType oldTensor,
                               ArrayRef<int64_t> newGridShape) {
  auto oldLayout = mlir::cast<ttcore::MetalLayoutAttr>(oldTensor.getEncoding());
  if (oldLayout.getGridShape(oldTensor) == newGridShape) {
    return oldTensor;
  }

  auto [newShape, reblockMap] = ttmlir::utils::calculateReblockMapForGrid(
      oldTensor.getShape(), newGridShape, oldTensor.getContext());

  ttcore::MetalLayoutAttr newLayout = oldLayout.withIndexAffineMap(reblockMap);
  return RankedTensorType::get(newShape, oldTensor.getElementType(), newLayout);
}

std::optional<SmallVector<int64_t>>
computeDimConstraints(mlir::ArrayRef<mlir::AffineMap> indexingMaps,
                      mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes) {
  TT_assert(!indexingMaps.empty());
  TT_assert(indexingMaps.size() == shapes.size());
  auto numDims = indexingMaps.front().getNumDims();
  SmallVector<int64_t> constrainedDims(numDims, 0);
  for (auto [shapeIdx, shape] : llvm::enumerate(shapes)) {
    auto dimProjectionMap =
        mlir::inverseAndBroadcastProjectedPermutation(indexingMaps[shapeIdx]);
    auto impliedDimConstraints = dimProjectionMap.compose(shape);

    for (auto [dimIdx, dimConstraint] :
         llvm::enumerate(impliedDimConstraints)) {
      if (dimConstraint == 0) {
        continue;
      }

      // Early exit if shapes are incompatible.
      if (constrainedDims[dimIdx] != 0 &&
          constrainedDims[dimIdx] != dimConstraint) {
        return std::nullopt;
      }
      constrainedDims[dimIdx] = dimConstraint;
    }
  }
  return constrainedDims;
}

} // namespace mlir::tt::d2m::utils
