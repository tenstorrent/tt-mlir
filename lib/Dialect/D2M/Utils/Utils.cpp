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

// Generates a permutation map that interleaves groups the corresponding grid
// and shard dimensions of a tensor.
// Example: (g0, g1, g2, s0, s1, s2) -> (g0, s0, g1, s1, g2, s2)
SmallVector<int64_t> createInterleavePermutationMask(int64_t rank) {
  TT_assertv(rank % 2 == 0, "Rank must be even");
  int64_t halfRank = rank / 2;
  SmallVector<int64_t> permutation(rank);
  for (int64_t i = 0; i < rank; ++i) {
    permutation[i] = (i % 2 * halfRank) + i / 2;
  }
  return permutation;
}

// Generates a permutation map that swaps the positions of the grid and shard
// dimensions of a tensor
// Example: (g0, s0, g1, s1, g2, s2) -> (g0, g1, g2, s0, s1, s2)
SmallVector<int64_t> createInverseInterleavePermutationMask(int64_t rank) {
  return ttmlir::utils::inversePermutation(
      createInterleavePermutationMask(rank));
}

SmallVector<int64_t> interleaveGridAndShardDims(ArrayRef<int64_t> shape) {
  return ttmlir::utils::applyPermutation(
      shape, createInterleavePermutationMask(shape.size()));
}

AffineMap deinterleaveMapDimsAndResults(AffineMap map) {
  // permute the dimension ordering to match the interleaved grid and shard dim
  // of the map
  auto dimPerm = createInterleavePermutationMask(map.getNumDims());
  auto dimSwapMap =
      mlir::AffineMap::getPermutationMap(dimPerm, map.getContext());
  auto dimSwappedMap = map.compose(dimSwapMap);

  // inverse permute the result ordering to match the original grid and shard
  // dim ordering
  auto invResultPermutation =
      createInverseInterleavePermutationMask(map.getNumResults());
  SmallVector<AffineExpr> results(map.getNumResults());
  for (int64_t i = 0; i < map.getNumResults(); ++i) {
    results[i] = dimSwappedMap.getResult(invResultPermutation[i]);
  }
  return mlir::AffineMap::get(map.getNumDims(), 0, results, map.getContext());
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

  // reorganize dims, pairing grid and shard dimensions together
  auto interleavedInputShape = detail::interleaveGridAndShardDims(inputShape);
  auto interleavedOutputShape = detail::interleaveGridAndShardDims(outputShape);

  // Construct a map that transforms input (grid x shard) to logical space.
  mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, ctx);
  auto stride = mlir::getAffineConstantExpr(1, ctx);
  for (int64_t i = outputRank - 1; i >= 0; --i) {
    auto dim = mlir::getAffineDimExpr(i, ctx);
    auto dimMod = dim % interleavedOutputShape[i];
    expr = (dimMod * stride) + expr;
    stride = stride * interleavedOutputShape[i];
  }
  auto outputToLogical = mlir::AffineMap::get(outputRank, 0, {expr}, ctx);

  // Construct a map that transforms the logical space to output (grid x shard).
  llvm::SmallVector<mlir::AffineExpr> toInputExprs;
  stride = mlir::getAffineConstantExpr(1, ctx);
  auto dim = mlir::getAffineDimExpr(0, ctx);
  for (int64_t i = inputRank - 1; i >= 0; --i) {
    toInputExprs.push_back((dim.floorDiv(stride)) % interleavedInputShape[i]);
    stride = stride * interleavedInputShape[i];
  }
  toInputExprs = llvm::to_vector(llvm::reverse(toInputExprs));
  auto logicalToInput = mlir::AffineMap::get(1, 0, toInputExprs, ctx);

  auto composeMap = logicalToInput.compose(outputToLogical);
  auto finalMap = detail::deinterleaveMapDimsAndResults(composeMap);
  finalMap = ttmlir::utils::simplifyZeroFloorDiv(finalMap);
  finalMap = ttmlir::utils::simplifyRedundantMod(finalMap, outputShape);
  return finalMap;
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

AffineMap concatInversePermutationMap(mlir::ArrayRef<AffineMap> affineMaps,
                                      bool reverse) {
  assert(!affineMaps.empty());
  auto *ctx = affineMaps.front().getContext();

  // Concat all of the indexing maps together, matmul example:
  // (d0, d1, d2) -> (d0, d2)
  // (d0, d1, d2) -> (d2, d1)
  // (d0, d1, d2) -> (d0, d1)
  // Becomes:
  // (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)
  AffineMap concat;

  // We typically want to reverse it so that output dimensions get priority for
  // the inverse permutation.
  if (reverse) {
    const auto reversedMaps = llvm::to_vector(llvm::reverse(affineMaps));
    concat = mlir::concatAffineMaps(reversedMaps, ctx);
  } else {
    concat = mlir::concatAffineMaps(affineMaps, ctx);
  }

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
    return getPhysicalTensorOrMemref(toLayoutOp.getInitOperand());

  } else if (auto genericOp = mlir::dyn_cast<mlir::tt::d2m::GenericOp>(
                 tensorOrMemref.getDefiningOp())) {
    // Assume that if the defining op is a generic op, the output is the first
    // of the outputs.
    physTensor = getPhysicalTensorOrMemref(genericOp.getOutputs()[0]);
  }

  return physTensor;
}

ArrayRef<int64_t> getGridShape(Value tensorOrMemref) {
  TT_assertv((mlir::isa<RankedTensorType>(tensorOrMemref.getType()) ||
              mlir::isa<MemRefType>(tensorOrMemref.getType())),
             "Expected a tensor or memref type");
  return ttcore::getDeviceLayout(tensorOrMemref)
      .getGridShape(mlir::cast<ShapedType>(tensorOrMemref.getType()));
}

} // namespace mlir::tt::d2m::utils
