// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_AFFINEMAPUTILS_H
#define TTMLIR_AFFINEMAPUTILS_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace ttmlir::utils {

/// Returns a new shape by applying `map` to the input shape.
template <typename Vector>
llvm::SmallVector<int64_t> evalShape(mlir::AffineMap map, Vector shape) {
  mlir::SmallVector<int64_t> lastIndex;
  for (auto dim : shape) {
    lastIndex.push_back(dim - 1);
  }

  auto result = map.compose(lastIndex);
  for (auto &dim : result) {
    dim += 1;
  }
  return result;
}

/// Returns a new affine map with all symbols replaced with given constant
/// values.
inline mlir::AffineMap
replaceAffineMapSymbols(mlir::AffineMap map, mlir::ArrayRef<int64_t> symbols) {
  TT_assertv(map.getNumSymbols() == symbols.size(),
             "Number of symbols must match number of replacement values");

  mlir::SmallVector<mlir::AffineExpr> symReplacements;
  for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
    symReplacements.push_back(
        getAffineConstantExpr(symbols[i], map.getContext()));
  }

  mlir::SmallVector<mlir::AffineExpr> dimReplacements;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dimReplacements.push_back(getAffineDimExpr(i, map.getContext()));
  }

  unsigned numResultSyms = 0;
  return map.replaceDimsAndSymbols(dimReplacements, symReplacements,
                                   map.getNumDims(), numResultSyms);
}

/// Generates an affine map translating ND grid + ND shard coordinates into ND
/// grid + linearized offset.
/// Example: strides=[4,2] -> (g0,g1,s0,s1) -> (g0,g1,4*s0+2*s1)
inline mlir::AffineMap
generateAffineMapFromShardStrides(mlir::ArrayRef<int64_t> strides,
                                  mlir::MLIRContext *context) {
  int64_t rank = strides.size();
  mlir::SmallVector<mlir::AffineExpr> mapExprs(rank + 1);

  for (int64_t i = 0; i < rank; i++) {
    mapExprs[i] = getAffineDimExpr(i, context);
  }

  mapExprs[rank] = getAffineConstantExpr(0, context);
  for (int64_t i = rank - 1; i >= 0; i--) {
    mlir::AffineExpr shardDim = getAffineDimExpr(rank + i, context);
    mlir::AffineExpr stride = getAffineConstantExpr(strides[i], context);
    mapExprs[rank] = shardDim * stride + mapExprs[rank];
  }

  auto map = mlir::AffineMap::get(strides.size() * 2, 0, mapExprs, context);
  return map;
}

/// Returns a new affine map by dropping the last N results of input map
inline mlir::AffineMap affineMapDropBackResults(mlir::AffineMap map,
                                                unsigned numResultsToDrop) {
  return map.dropResults(llvm::to_vector(llvm::seq<int64_t>(
      map.getNumResults() - numResultsToDrop, map.getNumResults())));
}

/// Returns a new affine map by taking just the first N results of input map
inline mlir::AffineMap affineMapTakeFrontResults(mlir::AffineMap map,
                                                 unsigned numResultsToTake) {
  TT_assert(numResultsToTake <= map.getNumResults());
  return map.dropResults(llvm::to_vector(
      llvm::seq<int64_t>(numResultsToTake, map.getNumResults())));
}

/// Returns a new affine map with only the selected result.
inline mlir::AffineMap affineMapSelectOneOutput(mlir::AffineMap map,
                                                unsigned selectedResult) {
  mlir::SmallVector<int64_t> dropMask;
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    if (i != selectedResult) {
      dropMask.push_back(i);
    }
  }
  return map.dropResults(mlir::ArrayRef<int64_t>(dropMask));
}

/// Applies an affine map to input values, returning an AffineApplyOp for each
/// result.
inline llvm::SmallVector<mlir::Value>
fullyApplyAffineMap(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::AffineMap map, mlir::ValueRange inputs) {
  llvm::SmallVector<mlir::Value> results;
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    results.push_back(builder.create<mlir::affine::AffineApplyOp>(
        loc, affineMapSelectOneOutput(map, i), inputs));
  }
  return results;
}

/// Derives a new grid shape by sampling an affine map over a reference grid
/// shape.
inline llvm::SmallVector<int64_t>
applyMapToGrid(mlir::ArrayRef<int64_t> gridShape, mlir::AffineMap map,
               bool assertResultStartsAtOrigin = true) {
  TT_assertv(gridShape.size() == map.getNumDims(),
             "Grid shape must have the same number of dimensions as the map");
  llvm::SmallVector<int64_t> lowerBound = llvm::SmallVector<int64_t>(
      map.getNumResults(), std::numeric_limits<int64_t>::max());
  llvm::SmallVector<int64_t> resultGridShape =
      llvm::SmallVector<int64_t>(map.getNumResults(), 0);
  ttmlir::utils::sample(gridShape, [&](llvm::SmallVector<int64_t, 8> point) {
    llvm::SmallVector<int64_t> virtualPoint = map.compose(point);
    for (size_t i = 0; i < virtualPoint.size(); ++i) {
      resultGridShape[i] = std::max(resultGridShape[i], virtualPoint[i] + 1);
      lowerBound[i] = std::min(lowerBound[i], virtualPoint[i]);
    }
  });
  if (assertResultStartsAtOrigin) {
    TT_assertv(llvm::all_of(lowerBound, [](int64_t x) { return x == 0; }),
               "Grid must start at origin");
  }
  return resultGridShape;
}

// Utility function to create an identity inverse map for grid virtualization
// Returns a map: (d0, d1) -> (0, d0, d1) where the first result is deviceIndex
inline mlir::AffineMap
createIdentityGridInverseMap(mlir::MLIRContext *context) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, context);
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, context);
  return mlir::AffineMap::get(2, 0, {zero, d0, d1}, context);
}

// Utility function to derive grid inverse map from a layout's index_map.
// Takes an index_map like (d0, d1, d2, d3) -> (d1, d0, d2, d3) and creates
// the grid inverse map (d0, d1) -> (0, d1, d0) that properly composes with
// the forward map for roundtrip consistency.
//
// The index_map encodes virtual-to-physical coordinate mapping. The grid
// portion (first gridRank results) may permute the grid dimensions. This
// function extracts that permutation and computes its inverse for use in
// the grid attribute.
inline mlir::AffineMap
createGridInverseMapFromIndexMap(mlir::AffineMap indexMap, unsigned gridRank,
                                 mlir::MLIRContext *context) {
  // If no index_map or it's empty/identity, return identity grid inverse map
  if (!indexMap || indexMap.isEmpty() || indexMap.isIdentity()) {
    return createIdentityGridInverseMap(context);
  }

  // Extract grid portion of the index_map (first gridRank results).
  // The index_map is (d0, d1, d2, d3) -> (results...) where the first
  // gridRank results correspond to grid coordinates.
  llvm::SmallVector<mlir::AffineExpr> gridResults;
  for (unsigned i = 0; i < gridRank; ++i) {
    gridResults.push_back(indexMap.getResult(i));
  }

  // Create a map with just the grid dimensions
  auto gridMap = mlir::AffineMap::get(gridRank, 0, gridResults, context);

  // Get the inverse permutation
  auto invGridMap = mlir::inversePermutation(gridMap);

  // If inverse is null (not a valid permutation), fall back to identity
  if (!invGridMap) {
    return createIdentityGridInverseMap(context);
  }

  // Build grid inverse map with device ID prefix: (d0, d1) -> (0, inv_y, inv_x)
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, context);
  llvm::SmallVector<mlir::AffineExpr> invResults;
  invResults.push_back(zero);
  for (auto result : invGridMap.getResults()) {
    invResults.push_back(result);
  }

  return mlir::AffineMap::get(gridRank, 0, invResults, context);
}

// Calculate a reblocking affine map from inputShape to outputShape.
inline mlir::AffineMap calculateReblockMap(mlir::ArrayRef<int64_t> inputShape,
                                           mlir::ArrayRef<int64_t> outputShape,
                                           mlir::MLIRContext *ctx) {
  TT_assert(utils::volume<int64_t>(inputShape) ==
            utils::volume<int64_t>(outputShape));
  int64_t inputRank = static_cast<int64_t>(inputShape.size());
  int64_t outputRank = static_cast<int64_t>(outputShape.size());
  TT_assertv(inputRank % 2 == 0, "Input rank must be even");
  TT_assertv(outputRank % 2 == 0, "Output rank must be even");

  if (inputShape == outputShape) {
    return mlir::AffineMap::getMultiDimIdentityMap(inputRank, ctx);
  }

  // Construct a map that transforms output (grid x shard) indices to row-major
  // flat indices.
  mlir::AffineExpr expr = mlir::getAffineConstantExpr(0, ctx);
  auto overallStride = mlir::getAffineConstantExpr(1, ctx);
  for (auto [i, dimStride] :
       utils::iterateInAscendingStrideOrder(outputShape)) {
    // Dims of size 1 contribute nothing.
    if (dimStride > 1) {
      auto dim = mlir::getAffineDimExpr(i, ctx);
      expr = dim * overallStride + expr;
      overallStride = overallStride * dimStride;
    }
  }
  auto outputToFlat = mlir::AffineMap::get(outputRank, 0, {expr}, ctx);

  // Construct a map that transforms flat indices to input (grid x shard)
  // indices.
  llvm::SmallVector<mlir::AffineExpr> toInputExprs(inputRank);
  overallStride = mlir::getAffineConstantExpr(1, ctx);
  auto dim = mlir::getAffineDimExpr(0, ctx);
  for (auto [i, dimStride] : utils::iterateInAscendingStrideOrder(inputShape)) {
    toInputExprs[i] = dim.floorDiv(overallStride);
    // Modulo on the outermost grid dim is unnecessary, but we allow "mod 1"
    // since it reduces the entire term to 0.
    if (!(i == 0 && dimStride != 1)) {
      toInputExprs[i] = toInputExprs[i] % dimStride;
    }
    overallStride = overallStride * dimStride;
  }
  auto flatToInput = mlir::AffineMap::get(1, 0, toInputExprs, ctx);

  return flatToInput.compose(outputToFlat);
}

/// Calculate a reblock affine map given a shape and new grid shape.
/// Returns the new tensor shape and the reblock affine map.
inline std::pair<mlir::SmallVector<int64_t>, mlir::AffineMap>
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

/// Concatenates the provided affine maps together and then inverts the map.
/// This is a convenient routine for deriving concrete iterator values.
///
/// Using matmul maps for example:
///   (d0, d1, d2) -> (d0, d2)
///   (d0, d1, d2) -> (d2, d1)
///   (d0, d1, d2) -> (d0, d1)
///
///   1. If reverse is set, it will reverse the provided affine maps first.
///   2. Concat all of the indexing maps together:
///        (d0, d1, d2) -> (d0, d1, d2, d1, d0, d2)
///   3. Invert the permutation, remapping the results to input iterators:
///        (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)
inline mlir::AffineMap
concatInversePermutationMap(llvm::SmallVector<mlir::AffineMap> affineMaps,
                            bool reverse) {
  assert(!affineMaps.empty());

  // Reverse the maps to give output dimensions priority in the inverse
  // permutation.
  if (reverse) {
    affineMaps = llvm::to_vector(llvm::reverse(affineMaps));
  }

  // Concatenate all indexing maps together.
  mlir::AffineMap concat =
      mlir::concatAffineMaps(affineMaps, affineMaps.front().getContext());

  // Invert the permutation to derive loop bounds from operand shapes.
  return mlir::inversePermutation(concat);
}

/// Build affine map from device indices to physical indices.
/// Reconstructs physical coordinates from grid + shard coordinates.
///
/// Example:
///   physical shape: [128, 256]
///   grid shape: [4, 8]
///   shard sizes: [32, 32]
///
///   Result: (d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)
///   where first 2 dims are grid coords, last 2 are shard coords.
inline mlir::AffineMap
buildDeviceToPhysicalMap(mlir::ArrayRef<int64_t> physicalShape,
                         mlir::ArrayRef<int64_t> gridShape,
                         mlir::MLIRContext *context) {
  assert(physicalShape.size() == gridShape.size() &&
         "Physical and grid must have same rank");

  size_t rank = physicalShape.size();
  mlir::SmallVector<mlir::AffineExpr> physicalExprs;
  physicalExprs.reserve(rank);

  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr gridDim = mlir::getAffineDimExpr(i, context);
    mlir::AffineExpr shardDim = mlir::getAffineDimExpr(rank + i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];

    physicalExprs.push_back(gridDim * shardSize + shardDim);
  }

  return mlir::AffineMap::get(rank * 2, 0, physicalExprs, context);
}

/// Build semi-affine map from physical indices to device indices.
/// Distributes the physical shape across a grid.
///
/// Example:
///   physical shape: [128, 256]
///   grid shape: [4, 8]
///
///   Result: (d0, d1) -> (d0 floordiv 32, d1 floordiv 32, d0 mod 32, d1 mod 32)
///   where shard sizes are [128/4=32, 256/8=32].
inline mlir::AffineMap
buildPhysicalToDeviceMap(mlir::ArrayRef<int64_t> physicalShape,
                         mlir::ArrayRef<int64_t> gridShape,
                         mlir::MLIRContext *context) {
  assert(physicalShape.size() == gridShape.size() &&
         "Physical and grid must have same rank");

  size_t rank = physicalShape.size();
  mlir::SmallVector<mlir::AffineExpr> deviceExprs;
  deviceExprs.reserve(rank * 2);

  // First rank results are grid coordinates.
  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = mlir::getAffineDimExpr(i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];
    deviceExprs.push_back(dim.floorDiv(shardSize));
  }

  // Next rank results are shard-local coordinates.
  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = mlir::getAffineDimExpr(i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];
    deviceExprs.push_back(dim % shardSize);
  }

  return mlir::AffineMap::get(rank, 0, deviceExprs, context);
}

inline std::optional<int64_t> getSumOfModuli(mlir::AffineExpr expr) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Add) {
      auto lhs = getSumOfModuli(binOp.getLHS());
      auto rhs = getSumOfModuli(binOp.getRHS());
      if (lhs.has_value() && rhs.has_value()) {
        return (lhs.value() + rhs.value());
      }
    } else if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        // We need to subtract 1, as the max value of the modulo op is one less
        // than the modulus value.
        return (rhsConst.getValue() - 1);
      }
    } else if (binOp.getKind() == mlir::AffineExprKind::Mul) {
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        auto lhs = getSumOfModuli(binOp.getLHS());
        if (lhs.has_value()) {
          return lhs.value() * rhsConst.getValue();
        }
      }
      if (auto lhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getLHS())) {
        auto rhs = getSumOfModuli(binOp.getRHS());
        if (rhs.has_value()) {
          return rhs.value() * lhsConst.getValue();
        }
      }
    }
  }
  return std::nullopt;
}

inline mlir::AffineExpr simplifyZeroFloorDivExpr(mlir::AffineExpr expr) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = simplifyZeroFloorDivExpr(binOp.getLHS());
    auto rhs = simplifyZeroFloorDivExpr(binOp.getRHS());

    if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
      if (auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        int64_t divisor = rhsConst.getValue();
        auto modSum = getSumOfModuli(lhs);
        if (modSum.has_value() && modSum.value() < divisor) {
          return mlir::getAffineConstantExpr(0, expr.getContext());
        }
      }
    }

    switch (binOp.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::Mod:
      return lhs % rhs;
    case mlir::AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case mlir::AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      return expr;
    }
  }
  return expr;
}

/// Simplifies the affine map by finding sub expressions in results that always
/// evaluate to zero.
/// Specifically, it looks for: ((dim0 mod M) + (dim1 mod N) ... ) floorDiv Q.
/// If sum(M, N, ...) <= Q, the expression is replaced with 0.
inline mlir::AffineMap simplifyZeroFloorDiv(mlir::AffineMap map) {
  mlir::SmallVector<mlir::AffineExpr> newResults;
  for (auto result : map.getResults()) {
    newResults.push_back(simplifyZeroFloorDivExpr(result));
  }
  return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                              map.getContext());
}

inline std::optional<int64_t>
getExprUpperBound(mlir::AffineExpr expr, mlir::ArrayRef<int64_t> dimBounds) {

  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() < dimBounds.size()) {
      return dimBounds[dimExpr.getPosition()] - 1;
    }
    return std::nullopt;
  }
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    // We conservatively return nullopt for negative constants to avoid
    // handling sign issues in Mul/Div for upper bound calculation.
    if (constExpr.getValue() < 0) {
      return std::nullopt;
    }
    return constExpr.getValue();
  }
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {

    auto lhs = getExprUpperBound(binOp.getLHS(), dimBounds);

    // Quick check for Mod with constant RHS
    if (binOp.getKind() == mlir::AffineExprKind::Mod) {

      auto lhs = getExprUpperBound(binOp.getLHS(), dimBounds);
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        int64_t rhsVal = rhsConst.getValue();
        if (rhsVal <= 0) {
          return std::nullopt;
        }
        if (lhs.has_value()) {
          auto r = std::min(lhs.value(), rhsVal - 1);
          return r;
        }
        return rhsVal - 1;
      }
      return std::nullopt;
    }

    if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        int64_t rhsVal = rhsConst.getValue();
        if (rhsVal <= 0) {
          return std::nullopt;
        }
        if (lhs.has_value()) {
          return lhs.value() / rhsVal;
        }
      }
      return std::nullopt;
    }

    auto rhs = getExprUpperBound(binOp.getRHS(), dimBounds);

    if (lhs.has_value() && rhs.has_value()) {
      switch (binOp.getKind()) {
      case mlir::AffineExprKind::Add: {
        return lhs.value() + rhs.value();
      }
      case mlir::AffineExprKind::Mul: {
        return lhs.value() * rhs.value();
      }
      default:
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

inline mlir::AffineExpr
simplifyRedundantModExpr(mlir::AffineExpr expr,
                         mlir::ArrayRef<int64_t> dimBounds) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = simplifyRedundantModExpr(binOp.getLHS(), dimBounds);
    auto rhs = simplifyRedundantModExpr(binOp.getRHS(), dimBounds);

    if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      if (auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        auto lhsUB = getExprUpperBound(lhs, dimBounds);
        if (lhsUB.has_value() && lhsUB.value() < rhsConst.getValue()) {
          return lhs;
        }
      }
      return lhs % rhs;
    }
    if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
      if (auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        auto lhsUB = getExprUpperBound(lhs, dimBounds);
        if (lhsUB.has_value() && lhsUB.value() < rhsConst.getValue()) {
          return mlir::getAffineConstantExpr(0, expr.getContext());
        }
      }
      return lhs.floorDiv(rhs);
    }

    switch (binOp.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case mlir::AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      return expr;
    }
  }
  return expr;
}

inline mlir::AffineMap simplifyRedundantMod(mlir::AffineMap map,
                                            mlir::ArrayRef<int64_t> dimBounds) {
  TT_assertv(map.getNumDims() == dimBounds.size(),
             "Number of dimension bounds must match number of map dimensions");
  mlir::SmallVector<mlir::AffineExpr> newResults;
  for (auto result : map.getResults()) {
    newResults.push_back(simplifyRedundantModExpr(result, dimBounds));
  }
  return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                              map.getContext());
}

inline mlir::AffineMap
simplifyAffineMapWithRangeAnalysis(mlir::AffineMap map,
                                   mlir::ArrayRef<int64_t> dimBounds) {
  return simplifyRedundantMod(simplifyZeroFloorDiv(map), dimBounds);
}

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPUTILS_H
