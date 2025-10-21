// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Utils.h"
#include "ttmlir/Asserts.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

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

Type getRegionLargestDstElemType(Region &region) {
  auto getTypeNumberOfBits = [](Type type) {
    return ttcore::getNumberOfBits(ttcore::elementTypeToDataType(type));
  };

  Type largestType = nullptr;
  region.walk([&](OperandLoadStoreRegisterOpInterface op) {
    // Only the typecast op has different input & output types, but it's a DST
    // in-place op so we simply check all the operands of all the compute ops.
    for (Value v : op.getOperation()->getOperands()) {
      Type t = ttcore::getOperandInnerElementType(v);

      if (!largestType ||
          (getTypeNumberOfBits(t) > getTypeNumberOfBits(largestType))) {
        largestType = t;
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

AffineMap concatInversePermutationMap(SmallVector<AffineMap> affineMaps,
                                      bool reverse) {
  assert(!affineMaps.empty());

  // We typically want to reverse it so that output dimensions get priority for
  // the inverse permutation.
  if (reverse) {
    affineMaps = llvm::to_vector(llvm::reverse(affineMaps));
  }

  // Concat all of the indexing maps together, matmul example:
  // (d0, d1, d2) -> (d0, d2)
  // (d0, d1, d2) -> (d2, d1)
  // (d0, d1, d2) -> (d0, d1)
  // Becomes:
  // (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)
  AffineMap concat =
      mlir::concatAffineMaps(affineMaps, affineMaps.front().getContext());

  // Invert the permutation to get a map that we can use to get the loop
  // bounds. Above example becomes: (d0, d1, d2, d3, d4, d5) -> (d0, d3, d1)
  return mlir::inversePermutation(concat);
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

AffineMap getCoreVirtualizationMap(mlir::Value tensorOrMemref) {
  auto physicalTensorOrMemref = getPhysicalTensorOrMemref(tensorOrMemref);
  auto shapedType = mlir::dyn_cast<ShapedType>(physicalTensorOrMemref.getType());
  TT_assertv(shapedType, "Expected a shaped type");

  auto layout = ttcore::getDeviceLayout(shapedType);

  if (auto shardLayout =
          mlir::dyn_cast_or_null<ttcore::ShardLayoutAttr>(layout)) {
    return shardLayout.getCoreVirtualizationMap();
  } else if (auto metalLayout =
                 mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(layout)) {
    // Core virtualization is stored in the IndexAffineMap field of MetalLayoutAttr
    auto map = metalLayout.getIndexAffineMap();

    // This is a hack to get around MetalLayoutAttr defaulting to an identity
    // map for indexAffineMap.
    return map.isIdentity() ? AffineMap{} : map;
  } else {
    return {};
  }
}

SmallVector<int64_t> getGridShape(mlir::Value tensorOrMemref) {
  auto shapedType = mlir::dyn_cast<ShapedType>(tensorOrMemref.getType());
  TT_assertv(shapedType, "Expected a shaped type");

  // Assume a default grid shape of {1, 1} if the layout is not found.
  auto layout = ttcore::getDeviceLayout(shapedType);
  return (layout) ? llvm::SmallVector<int64_t>(layout.getGridShape(shapedType))
                  : llvm::SmallVector<int64_t>({1, 1});
}

mlir::SmallVector<int64_t> applyMapToGrid(mlir::ArrayRef<int64_t> gridShape,
                                          mlir::AffineMap map) {
  if (!map || map.isIdentity()) {
    return SmallVector<int64_t>(gridShape.begin(), gridShape.end());
  }

  SmallVector<int64_t> resultGridShape =
      SmallVector<int64_t>(map.getNumResults(), 0);
  TT_assertv(gridShape.size() == map.getNumDims(),
             "Grid shape must have the same number of dimensions as the map");
  ttmlir::utils::sample(gridShape, [&](SmallVector<int64_t, 8> point) {
    SmallVector<int64_t> virtualPoint = map.compose(point);
    for (size_t i = 0; i < virtualPoint.size(); ++i) {
      resultGridShape[i] =
          std::max(resultGridShape[i], virtualPoint[i] + 1);
    }
  });
  return resultGridShape;
}

SmallVector<int64_t> getPhysicalGridShape(mlir::Value tensorOrMemref) {
  auto physTensorOrMemref = getPhysicalTensorOrMemref(tensorOrMemref);
  auto gridShape = getGridShape(physTensorOrMemref);
  return gridShape;
}

} // namespace mlir::tt::d2m::utils
