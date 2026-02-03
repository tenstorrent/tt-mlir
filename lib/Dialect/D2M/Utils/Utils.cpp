// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AffineExpr.h"

namespace mlir::tt::d2m::utils {

llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  const int64_t minGridValue = *llvm::min_element(targetGridShape);

  llvm::SmallVector<int64_t, 2> squareGrid(targetGridShape.size(),
                                           minGridValue);
  return squareGrid;
}

// Helper to find the largest DST element type in a region.
// Returns nullptr if no DST-using ops are found.
static Type findLargestDstElemType(Region &region) {
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

  return largestType;
}

Type getRegionLargestDstElemType(Region &region) {
  Type largestType = findLargestDstElemType(region);
  assert(largestType);
  auto getTypeNumberOfBits = [](Type type) {
    return ttcore::getNumberOfBits(ttcore::elementTypeToDataType(type));
  };
  TT_assert(getTypeNumberOfBits(largestType) <= 32u);
  return largestType;
}

Type getRegionLargestDstElemTypeOrNull(Region &region) {
  return findLargestDstElemType(region);
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

SmallVector<Value> buildGridIndices(OpBuilder &builder, Location loc,
                                    AffineMap indexingMap) {
  // Create dimension values by creating BlockIndexOp for each dimension
  SmallVector<Value> dimValues;
  for (unsigned i = 0; i < indexingMap.getNumDims(); ++i) {
    dimValues.push_back(
        builder.create<BlockIndexOp>(loc, static_cast<int64_t>(i)));
  }

  // For each result expression, use expandAffineExpr to translate to arith ops
  SmallVector<Value> indices;
  for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
    AffineExpr expr = indexingMap.getResult(i);
    Value result = mlir::affine::expandAffineExpr(builder, loc, expr, dimValues,
                                                  /*symbolValues=*/{});
    indices.push_back(result);
  }

  TT_assert(indices.size() == indexingMap.getNumResults());
  return indices;
}

} // namespace mlir::tt::d2m::utils
