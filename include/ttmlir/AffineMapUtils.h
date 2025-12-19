// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_AFFINEMAPUTILS_H
#define TTMLIR_AFFINEMAPUTILS_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/DenseSet.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

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
simplifyAffineExprWithRangeAnalysis(mlir::AffineExpr expr,
                                    mlir::ArrayRef<int64_t> dimBounds) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = simplifyAffineExprWithRangeAnalysis(binOp.getLHS(), dimBounds);
    auto rhs = simplifyAffineExprWithRangeAnalysis(binOp.getRHS(), dimBounds);

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

/// Simplifies an affine map by eliminating redundant mod/floordiv operations
/// using range analysis. If the upper bound of an operand is less than the
/// modulus, the mod is removed; if less than the divisor, floordiv becomes 0.
/// When performEquivalenceCheck is true, samples the domain to verify the
/// simplified map is equivalent; returns the original map if not.
inline mlir::AffineMap
simplifyAffineMapWithRangeAnalysis(mlir::AffineMap map,
                                   mlir::ArrayRef<int64_t> dimBounds,
                                   bool performEquivalenceCheck = true) {
  TT_assertv(map.getNumDims() == dimBounds.size(),
             "Number of dimension bounds must match number of map dimensions");
  mlir::SmallVector<mlir::AffineExpr> newResults;
  for (auto result : map.getResults()) {
    newResults.push_back(
        simplifyAffineExprWithRangeAnalysis(result, dimBounds));
  }
  auto simplifiedMap = mlir::AffineMap::get(
      map.getNumDims(), map.getNumSymbols(), newResults, map.getContext());

  // Roundtrip equivalence check: sample both maps over the entire domain
  // and verify they produce identical results at every point.
  if (performEquivalenceCheck) {
    bool equivalent = true;
    sample(dimBounds, [&](llvm::ArrayRef<int64_t> point) {
      if (map.compose(point) != simplifiedMap.compose(point)) {
        equivalent = false;
      }
    });
    if (!equivalent) {
      return map;
    }
  }

  return simplifiedMap;
}

inline mlir::AffineExpr isBinaryAdd(mlir::AffineExpr expr) {
  auto lhsBinExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  return (lhsBinExpr && lhsBinExpr.getKind() == mlir::AffineExprKind::Add)
             ? lhsBinExpr
             : mlir::AffineExpr{};
}

inline std::optional<mlir::AffineBinaryOpExpr>
isBinaryMul(mlir::AffineExpr expr) {
  auto lhsBinExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  return (lhsBinExpr && lhsBinExpr.getKind() == mlir::AffineExprKind::Mul)
             ? std::optional<mlir::AffineBinaryOpExpr>(lhsBinExpr)
             : std::nullopt;
}

inline std::optional<mlir::AffineDimExpr> isDimExpr(mlir::AffineExpr expr) {
  auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr);
  return dimExpr ? std::optional<mlir::AffineDimExpr>(dimExpr) : std::nullopt;
}

inline void
collectSumOperandsImpl(mlir::AffineExpr expr,
                       llvm::SmallVectorImpl<mlir::AffineExpr> &results) {
  // Add operations have least precedence, so collecting all operands
  // commutatively gathers the entire expressions; everything else must be a
  // child expr of the top-level add
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Add) {
      collectSumOperandsImpl(binOp.getLHS(), results);
      collectSumOperandsImpl(binOp.getRHS(), results);
      return;
    }
  }
  results.push_back(expr);
}

inline mlir::SmallVector<mlir::AffineExpr>
collectSumOperands(mlir::AffineExpr expr) {
  mlir::SmallVector<mlir::AffineExpr> ops;
  collectSumOperandsImpl(expr, ops);
  return ops;
}

// This function simplifies expressions of the form: (dimN * A +
// dimN * B + ...) mod C where some of the dims are assumed to be set to any
// constant value with known bounds. The max value of constDim * A is bounded
// by (dimBound - 1) * A, which allows other analysis to prove the modulus is
// redundant if some dims are constant.
inline std::pair<unsigned, mlir::AffineExpr> determineWorstCaseLHSValue(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &constDimBounds,
    mlir::AffineConstantExpr modRHS, mlir::MLIRContext *ctx) {

  std::optional<mlir::AffineDimExpr> dimOp;
  int64_t multiplier = 1;

  // Determine const multiplier on dim expr (if it exists)
  if (auto plainDimOp = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    dimOp = plainDimOp;
    multiplier = 1;
  } else if (auto mulOp = isBinaryMul(expr)) {
    // Handle Mul ops
    mlir::AffineExpr lhs = mulOp->getLHS(), rhs = mulOp->getRHS();

    // Get dim and const operands (could be in either order)
    auto lhsDim = llvm::dyn_cast<mlir::AffineDimExpr>(lhs);
    auto constOp = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    if (!lhsDim) {
      lhsDim = llvm::dyn_cast<mlir::AffineDimExpr>(rhs);
      constOp = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs);
    }

    if (!lhsDim || !constOp) {
      return {-1, expr};
    }
    dimOp = lhsDim;
    multiplier = constOp.getValue();
  } else {
    // return all other expr unchanged
    return {-1, expr};
  }

  if (auto bound = constDimBounds.lookup(dimOp->getPosition())) {
    int64_t dimMax = (bound - 1);
    int64_t maxValue = 0;

    auto modRHSValue = modRHS.getValue();
    if (dimMax * multiplier >= modRHSValue) {
      int64_t lcm = std::lcm(modRHSValue, multiplier);
      int64_t k = lcm / multiplier;
      int64_t maxRem = 0;
      for (int64_t i = 0; i < k; i++) {
        int64_t value = i * multiplier;
        int64_t rem = value % modRHSValue;
        maxRem = std::max(maxRem, rem);
      }
      maxValue = maxRem;
    } else {
      maxValue = dimMax * multiplier;
    }
    return {dimOp->getPosition(), mlir::getAffineConstantExpr(maxValue, ctx)};
  }
  return {-1, expr};
}

/// Calculates the coalescing factor for an affine map by sampling over the
/// given input shape. The coalescing factor is the greatest common divisor of
/// all contiguous run lengths, representing the maximum number of elements that
/// can be transferred in a single coalesced operation.
///
/// When numGridDims > 0, the first numGridDims dimensions are treated as "grid"
/// dimensions. For each grid coordinate, a shard coalescing factor is computed
/// by sampling over the remaining "shard" dimensions. The combined coalescing
/// factor is the GCD of all shard coalescing factors across all grid points.
///
/// This is useful for determining how to break up DMA transfers - a coalescing
/// factor equal to the shard volume means fully contiguous access within
/// shards, while a factor of 1 means each element must be transferred
/// individually.
///
/// Example: For a row-major 2D layout map (d0, d1) -> (d0 * 4 + d1)
///          with shape [2, 4], stride 1, and numGridDims=0, returns 8
///          (fully contiguous) because consecutive elements produce
///          addresses: 0,1,2,3,4,5,6,7.
///
/// Example: For a column-major map (d0, d1) -> (d1 * 2 + d0)
///          with shape [2, 4], stride 1, and numGridDims=0, returns 1
///          because consecutive row-major indices produce addresses:
///          0,2,4,6,1,3,5,7 (no consecutive runs longer than 1).
inline size_t calculateCoalescingFactor(mlir::AffineMap map,
                                        mlir::ArrayRef<int64_t> shape,
                                        int64_t stride,
                                        unsigned numGridDims = 0) {
  TT_assertv(map.getNumDims() == shape.size(),
             "Map dimensions must match shape size");
  TT_assertv(numGridDims <= shape.size(),
             "Number of grid dims cannot exceed shape size");

  // Extract grid and shard shapes
  mlir::ArrayRef<int64_t> gridShape = shape.take_front(numGridDims);
  mlir::ArrayRef<int64_t> shardShape = shape.drop_front(numGridDims);

  // If no shard dims, trivially contiguous (volume is 1)
  if (shardShape.empty()) {
    return 1;
  }

  size_t shardVolume = volume(shardShape);
  size_t combinedCoalescingFactor = shardVolume;

  TT_assertv(!gridShape.empty(), "Grid shape cannot be empty");

  sample(gridShape, [&](llvm::ArrayRef<int64_t> gridIndex) {
    if (combinedCoalescingFactor == 1) {
      return; // Can't decrease further
    }

    size_t shardCoalescingFactor = shardVolume;
    size_t currentRunLength = 0;
    llvm::SmallVector<int64_t, 4> expectedAddress;

    sample(shardShape, [&](llvm::ArrayRef<int64_t> shardIndex) {
      if (shardCoalescingFactor == 1) {
        return;
      }

      // Combine grid and shard indices
      llvm::SmallVector<int64_t> fullIndex;
      fullIndex.append(gridIndex.begin(), gridIndex.end());
      fullIndex.append(shardIndex.begin(), shardIndex.end());

      llvm::SmallVector<int64_t, 4> address = map.compose(fullIndex);
      if (expectedAddress.empty() || expectedAddress == address) {
        ++currentRunLength;
      } else {
        shardCoalescingFactor =
            std::gcd(shardCoalescingFactor, currentRunLength);
        currentRunLength = 1;
      }

      expectedAddress = address;
      expectedAddress.back() += stride;
    });

    // Account for final run
    shardCoalescingFactor = std::gcd(shardCoalescingFactor, currentRunLength);

    // Merge into combined factor
    combinedCoalescingFactor =
        std::gcd(combinedCoalescingFactor, shardCoalescingFactor);
  });

  return combinedCoalescingFactor;
}

/// Finds the minimum gap to the next target boundary for any achievable sum.
/// For floor division semantics: gap = target - (sum % target), or target if
/// sum is exactly on a boundary.
/// Returns the minimum such gap across all achievable sums.
inline int64_t minimizeGap(int64_t target, llvm::ArrayRef<int64_t> multipliers,
                           llvm::ArrayRef<int64_t> bounds,
                           int64_t constOffset) {
  assert(multipliers.size() == bounds.size());

  // Compute all achievable sums using dynamic programming.
  // Start with just the constant offset.
  llvm::DenseSet<int64_t> achievableSums;
  achievableSums.insert(constOffset);

  for (size_t i = 0; i < multipliers.size(); ++i) {
    llvm::DenseSet<int64_t> newSums;
    for (int64_t sum : achievableSums) {
      for (int64_t n = 0; n <= bounds[i]; ++n) {
        newSums.insert(sum + multipliers[i] * n);
      }
    }
    achievableSums = std::move(newSums);
  }

  // Find the minimum gap to the next boundary for any achievable sum.
  // For floor division: gap = target - (sum % target)
  // If sum % target == 0, we're at an exact multiple, so gap = target
  // (need full cycle to reach the next boundary).
  int64_t bestGap = target;

  for (int64_t sum : achievableSums) {
    int64_t remainder = sum % target;
    int64_t gap = (remainder == 0) ? target : (target - remainder);
    if (gap < bestGap) {
      bestGap = gap;
    }
    // Early exit if gap is 1 (best possible non-exact alignment)
    if (bestGap == 1) {
      break;
    }
  }

  return bestGap;
}

/// Helper function to recursively analyze contiguity of an affine expression.
/// This identifies the maximum coalescable run by analyzing mod/floordiv
/// expressions.
///
/// @param expr The expression to analyze
/// @param dimBounds Map of dimension positions to their bounds (number of
/// elements)
/// @param dimPos The target dimension being analyzed for contiguity
/// @param parentModulus Optional bound from enclosing mod/floordiv operation.
///                      When set, changes how expressions are evaluated.
/// @return -1 for unconstrained (no contiguity limit),
///         1 for error condition (analysis failed, no coalescing possible),
///         positive value for the contiguity bound
inline int64_t analyzeShardResultExprForContiguity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos, std::optional<int64_t> parentModulus = std::nullopt) {
  // Helper to check if steps exceed dim bounds
  auto checkBoundsAndReturn = [&](int64_t stepsNeeded) -> int64_t {
    int64_t targetDimBound = dimBounds.lookup(dimPos);
    if (stepsNeeded >= targetDimBound) {
      return -1; // Unconstrained - always contiguous within bounds
    }
    return stepsNeeded;
  };
  // Handle constant expressions
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    if (constExpr.getValue() < 0) {
      return 1; // Error: negative constant
    }
    return -1; // No contiguity constraint
  }

  // Handle dim expressions
  if (llvm::isa<mlir::AffineDimExpr>(expr)) {
    return -1; // No contiguity constraint
  }

  // Handle binary operations
  auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binOp) {
    return 1;
  }

  mlir::AffineExpr lhs = binOp.getLHS();
  mlir::AffineExpr rhs = binOp.getRHS();

  switch (binOp.getKind()) {
  case mlir::AffineExprKind::Add: {
    // Analyze each operand recursively, propagating rhsConst
    int64_t lhsBound = analyzeShardResultExprForContiguity(
        lhs, dimBounds, dimPos, parentModulus);
    int64_t rhsBound = analyzeShardResultExprForContiguity(
        rhs, dimBounds, dimPos, parentModulus);

    // Return GCD of bounds; -1 means unconstrained
    if (lhsBound == -1) {
      return rhsBound;
    }
    if (rhsBound == -1) {
      return lhsBound;
    }
    return std::gcd(lhsBound, rhsBound);
  }

  case mlir::AffineExprKind::Mul: {
    // Mul must have a constant operand
    auto lhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs);
    auto rhsConstExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);

    if (!lhsConst && !rhsConstExpr) {
      return 1; // Error: mul without constant operand
    }

    int64_t constVal =
        rhsConstExpr ? rhsConstExpr.getValue() : lhsConst.getValue();

    if (constVal < 0) {
      return 1; // Error: negative constant
    }

    int64_t lhsBound = analyzeShardResultExprForContiguity(
        lhs, dimBounds, dimPos, parentModulus);

    // if lhs is constrained, propagate the constraint up
    if (lhsBound != -1) {
      return lhsBound;
    }

    // If rhsConst is not set, the contiguity bound is unconstrained
    if (!parentModulus.has_value()) {
      return -1;
    }

    // If rhsConst % constVal == 0, bound is rhsConst / constVal
    if (*parentModulus % constVal == 0) {
      return *parentModulus / constVal;
    }
    // If rhsConst % constVal != 0, return error
    return 1;
  }

  case mlir::AffineExprKind::Mod:
  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    // RHS must be constant
    auto rhsConstExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    assert(rhsConstExpr && "Mod/FloorDiv RHS must be constant expression");

    int64_t modulus = rhsConstExpr.getValue();
    if (modulus <= 0) {
      return 1;
    }

    // Check if LHS is also floordiv/mod - error condition
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs)) {
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Mod ||
          lhsBinOp.getKind() == mlir::AffineExprKind::FloorDiv) {
        return 1;
      }
    }

    // If LHS is plain dim expr, check if it's the target dim
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
      if (dimExpr.getPosition() == dimPos) {
        return checkBoundsAndReturn(modulus);
      }
      // Different dim - unconstrained for target dim
      return -1;
    }

    // If LHS is a Mul, return result from calling func on lhs with boundValue
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs)) {
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Mul) {
        return analyzeShardResultExprForContiguity(lhs, dimBounds, dimPos,
                                                   modulus);
      }

      // If LHS is an Add, find worst-case gap using minimizeGap algorithm
      // This handles patterns like (A*d0 + B*d1 + ... + C) mod N
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Add) {
        auto sumOperands = collectSumOperands(lhs);

        // Track target dim separately from other dims
        int64_t targetMultiplier = 0;
        bool foundTargetDim = false;
        // Track (dimPosition, multiplier) pairs for other dimensions
        llvm::SmallVector<std::pair<unsigned, int64_t>> otherDimInfo;
        int64_t constSum = 0;

        for (auto sumOperand : sumOperands) {
          // Case 0: constants - accumulate into constSum
          if (auto constExpr =
                  llvm::dyn_cast<mlir::AffineConstantExpr>(sumOperand)) {
            constSum += constExpr.getValue();
            continue;
          }

          // Case 1: Plain dim expression (multiplier = 1)
          if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(sumOperand)) {
            if (dimExpr.getPosition() == dimPos) {
              if (foundTargetDim) {
                return 1; // Multiple occurrences of target dim, fallback
              }
              foundTargetDim = true;
              targetMultiplier = 1;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), 1});
            }
            continue;
          }

          // Case 2: Mul expression (dim * const)
          if (auto mulOp = isBinaryMul(sumOperand)) {
            auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(mulOp->getLHS());
            auto constExpr =
                llvm::dyn_cast<mlir::AffineConstantExpr>(mulOp->getRHS());

            if (!dimExpr || !constExpr) {
              return 1; // Cannot analyze, fallback
            }

            int64_t mulValue = constExpr.getValue();

            if (dimExpr.getPosition() == dimPos) {
              if (foundTargetDim) {
                return 1; // Multiple occurrences of target dim, fallback
              }
              foundTargetDim = true;
              targetMultiplier = mulValue;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), mulValue});
            }
            continue;
          }

          // Cannot analyze other patterns
          return 1;
        }

        // If we found the target dim, compute worst-case contiguity
        if (foundTargetDim && targetMultiplier > 0) {
          if (constSum < 0) {
            return 1; // Negative constants not analyzed, fallback
          }

          int64_t gap;
          if (otherDimInfo.empty()) {
            // Simple case: just the target dim and a constant offset
            // Gap is distance from constSum to the next modulus boundary
            int64_t remainder = constSum % modulus;
            gap = (remainder == 0) ? modulus : (modulus - remainder);
          } else {
            // Build multipliers and bounds arrays from tracked dimension info
            llvm::SmallVector<int64_t> otherMultipliers;
            llvm::SmallVector<int64_t> otherBounds;
            for (auto [dimPosition, mul] : otherDimInfo) {
              otherMultipliers.push_back(mul);
              // Bound is (dimSize - 1) since dim ranges from 0 to dimSize-1
              otherBounds.push_back(dimBounds.lookup(dimPosition) - 1);
            }

            // Use minimizeGap to find the best alignment (minimum gap)
            gap = minimizeGap(modulus, otherMultipliers, otherBounds, constSum);
          }

          int64_t stepsNeeded = (gap + targetMultiplier - 1) /
                                targetMultiplier; // Ceiling division
          return checkBoundsAndReturn(stepsNeeded);
        }

        // Target dim not found in expression - unconstrained
        return -1;
      }
    }

    // Default: recursively analyze LHS with boundValue as the new constraint
    return analyzeShardResultExprForContiguity(lhs, dimBounds, dimPos, modulus);
  }

  default:
    return 1;
  }
}

/// Analyzes the stride (coefficient) of a specific dimension in an affine
/// expression. The stride represents how much the output changes when the
/// dimension increases by 1.
///
/// Returns:
///   -1: Error/cannot analyze the expression
///   0: Dimension doesn't affect this expression
///   >0: The stride (coefficient) of the dimension
///
/// Examples:
/// - d0 -> stride is 1
/// - d0 * 128 -> stride is 128
/// - (d0 * 64) mod 4096 -> stride is 64 (mod doesn't change stride)
/// - d0 * 128 + (d0 * 64) mod 4096 -> stride is 128 + 64 = 192
/// - (d0 * 16) floorDiv 2 -> stride is 16 / 2 = 8
/// - (d0 * 2) floorDiv 16 -> stride is 0 (output changes less than once per
/// step)
inline int64_t analyzeExprForDimStride(mlir::AffineExpr expr, unsigned dimPos) {
  // Constant: stride = 0 (doesn't depend on dim)
  if (llvm::isa<mlir::AffineConstantExpr>(expr)) {
    return 0;
  }

  // Dim expression
  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    return dimExpr.getPosition() == dimPos ? 1 : 0;
  }

  auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binOp) {
    return -1; // Cannot analyze
  }

  auto lhs = binOp.getLHS();
  auto rhs = binOp.getRHS();

  switch (binOp.getKind()) {
  case mlir::AffineExprKind::Add: {
    int64_t lhsStride = analyzeExprForDimStride(lhs, dimPos);
    int64_t rhsStride = analyzeExprForDimStride(rhs, dimPos);
    if (lhsStride < 0 || rhsStride < 0) {
      return -1;
    }
    return lhsStride + rhsStride;
  }

  case mlir::AffineExprKind::Mul: {
    auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    // To simplify analysis, assert that rhs is always a constant here
    TT_assertv(
        rhsConst,
        "analyzeExprForDimStride: analysis expects MulOp rhs to be a constant");
    int64_t lhsStride = analyzeExprForDimStride(lhs, dimPos);
    if (lhsStride < 0) {
      return -1;
    }
    return lhsStride * rhsConst.getValue();
  }

  case mlir::AffineExprKind::Mod: {
    // Mod doesn't change stride, it just limits the range
    return analyzeExprForDimStride(lhs, dimPos);
  }

  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    if (!rhsConst || rhsConst.getValue() <= 0) {
      return -1;
    }

    int64_t lhsStride = analyzeExprForDimStride(lhs, dimPos);
    if (lhsStride < 0) {
      return -1;
    }

    int64_t divisor = rhsConst.getValue();
    // If lhsStride >= divisor, output changes by lhsStride/divisor per step
    // If lhsStride < divisor, output changes less than once per step (stride 0)
    // return std::max(1l, lhsStride / divisor);
    return std::max(1l, lhsStride / divisor);
  }

  default:
    return -1;
  }
}

/// Analyzes stride alignment for shard dimensions in an affine map.
/// For contiguous access (stride=1), the strides must satisfy:
/// - Innermost shard dimension has stride 1
/// - Each outer shard dimension has stride = inner_stride * inner_bound
///
/// @param map The affine map to analyze
/// @param shape The shape providing bounds for each dimension
/// @param numGridDims The number of grid dimensions (shard dims start after)
/// @param numGridResults The number of grid results in the map
/// @return Coalescing factor limited by stride misalignment:
///         - std::nullopt: Strides are properly aligned (no constraint)
///         - Positive value: Maximum coalescing factor due to stride issues
inline int64_t analyzeShardDimStrides(mlir::AffineMap map,
                                      mlir::ArrayRef<int64_t> shape,
                                      unsigned numGridDims,
                                      unsigned numGridResults) {
  int numShardDims = shape.size() - numGridDims;
  TT_assert(numShardDims > 0);
  // TODO: fix this
  TT_assert(numGridResults == map.getNumResults() - 1);

  mlir::AffineExpr shardResult = map.getResult(numGridResults);
  size_t innermostDimPos = map.getNumDims() - 1;
  int64_t innermostStride =
      analyzeExprForDimStride(shardResult, innermostDimPos);
  if (innermostStride < 0) {
    return 1;
  }

  int64_t accumulatedStride = innermostStride * shape[innermostDimPos];
  llvm::dbgs() << "\n[analyzeShardDimStrides] shape: "
               << ttmlir::utils::formatIterable(shape, "x") << "\n";
  llvm::dbgs() << "\n[analyzeShardDimStrides] expr: " << shardResult << "\n";
  for (int i = map.getNumDims() - 2; i >= static_cast<int>(numGridDims); --i) {
    unsigned dimPos = i;
    if (shape[dimPos] == 1) {
      // for unit dims, skip over
      llvm::dbgs() << "    [analyzeShardDimStrides] skipping unit dim @ d"
                   << dimPos << "\n";
      continue;
    }
    int64_t stride = analyzeExprForDimStride(shardResult, dimPos);
    llvm::dbgs() << "    [analyzeShardDimStrides] stride @ d" << dimPos << " = "
                 << stride << " (acc = " << accumulatedStride << ")\n";
    if (stride < 0) {
      return 1;
    }
    if (stride != accumulatedStride) {
      llvm::dbgs() << "      [analyzeShardDimStrides] stride mismatch @ d"
                   << dimPos << ": " << stride << " != " << accumulatedStride
                   << "\n";
      break;
    }
    accumulatedStride = stride * shape[dimPos];
  }

  return accumulatedStride;
}

inline bool exprContainsDim(mlir::AffineExpr expr, unsigned dimPos) {
  bool foundDim = false;
  expr.walk([&](mlir::AffineExpr e) {
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(e)) {
      if (dimExpr.getPosition() == dimPos) {
        foundDim = true;
      }
    }
  });
  return foundDim;
}

inline bool exprIsSpecificDimExpr(mlir::AffineExpr expr, unsigned dimPos) {
  return llvm::isa<mlir::AffineDimExpr>(expr) &&
         llvm::dyn_cast<mlir::AffineDimExpr>(expr).getPosition() == dimPos;
}

/// Analyzes an affine expression to find the minimum change in dimension
/// `dimPos` needed to change the output value. This is used for grid dimension
/// analysis where any change in grid indexing is a discontinuity.
///
/// Precondition: The expression must contain dimension `dimPos`.
///
/// Examples:
/// - `d0` -> 1 (every step changes output)
/// - `d0 floorDiv N` -> N (need N steps to change output)
/// - `(d0 + C) floorDiv N` -> N - C (steps until crossing divisor boundary)
/// - `(M*d0 + C) floorDiv N` -> ceil((N - C) / M)
///
/// @return Positive value: minimum change in dimPos needed to change output
///         -1: expression is unconstrained (dimPos doesn't affect this result)
///         1: conservative fallback when expression cannot be analyzed
inline int64_t analyzeGridResultExprForDiscontinuity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos) {
  // Handle constant expressions - unconstrained (dim doesn't affect output)
  if (llvm::isa<mlir::AffineConstantExpr>(expr)) {
    return -1;
  }

  // Handle dim expressions
  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() == dimPos) {
      return 1; // Direct dim reference, any change causes output change
    }
    return -1; // Different dim, unconstrained
  }

  auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binOp) {
    return 1; // Cannot analyze, conservative fallback
  }

  mlir::AffineExpr lhs = binOp.getLHS();
  mlir::AffineExpr rhs = binOp.getRHS();

  switch (binOp.getKind()) {
  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    if (!rhsConst || rhsConst.getValue() <= 0) {
      return 1; // Cannot analyze, conservative fallback
    }
    int64_t divisor = rhsConst.getValue();

    // Extract multiplier and offset from LHS: looking for M*d + C pattern
    int64_t multiplier = 0;
    bool foundDim = false;

    // Helper to check if steps exceed dim bounds
    auto checkBoundsAndReturn = [&](int64_t stepsNeeded) -> int64_t {
      int64_t targetDimBound = dimBounds.lookup(dimPos);
      if (stepsNeeded >= targetDimBound) {
        return -1; // Unconstrained - always contiguous within bounds
      }
      return stepsNeeded;
    };

    // Pattern 1: d_dimPos floorDiv N -> need N steps to change output
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
      if (dimExpr.getPosition() == dimPos) {
        return checkBoundsAndReturn(divisor);
      } else {
        // unrelated dim by itself poses no constraints
        return -1;
      }
    }

    if (auto lhsMod = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(lhs);
        lhsMod && lhsMod.getKind() == mlir::AffineExprKind::Mod) {

      // strip away mod, it doesn't impact analysis
      mlir::AffineExpr fusedExpr;
      if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
        fusedExpr = lhsMod.getLHS().floorDiv(divisor);
      } else {
        fusedExpr = lhsMod.getLHS().ceilDiv(divisor);
      }

      return analyzeGridResultExprForDiscontinuity(fusedExpr, dimBounds,
                                                   dimPos);
    }

    // Pattern 2: (M * d_dimPos) floorDiv N -> need ceil(N/M) steps
    if (auto maybeMul = isBinaryMul(lhs)) {
      auto mulRhsConst =
          llvm::dyn_cast<mlir::AffineConstantExpr>(maybeMul.value().getRHS());
      if (!mulRhsConst) {
        return 1; // Mul RHS must be constant, cannot analyze otherwise
      }
      int64_t mulValue = mulRhsConst.getValue();
      // Now we know RHS is a constant, just check LHS for dim
      if (exprIsSpecificDimExpr(maybeMul.value().getLHS(), dimPos)) {
        // ceil(divisor / mulValue)
        int64_t stepsNeeded = (divisor + mulValue - 1) / mulValue;
        return checkBoundsAndReturn(stepsNeeded);
      } else {
        return -1;
      }
    }

    // Pattern 3: (A*d0 + B*d1 + ...) floorDiv N
    if (!foundDim) {
      if (auto maybeAdd = isBinaryAdd(lhs)) {
        auto sumOperands = collectSumOperands(maybeAdd);

        // if the expression does not contain the target dim, it's unconstrained
        if (!exprContainsDim(lhs, dimPos)) {
          return -1;
        }

        mlir::AffineExpr targetDimExpr;
        // Track (dimPosition, multiplier) pairs for other dimensions
        llvm::SmallVector<std::pair<unsigned, int64_t>> otherDimInfo;
        int64_t constSum = 0;

        // Case: ((targetDim floorDiv N) + unrelated dim crap) floorDiv M) ...
        bool foundNestedFloordiv = false;
        bool foundNonDivExprWithTargetDim = false;
        int64_t fusedDivResult = 0;
        for (auto sumOperand : sumOperands) {
          if (auto lhsFloordiv =
                  mlir::dyn_cast<mlir::AffineBinaryOpExpr>(sumOperand);
              lhsFloordiv &&
              (lhsFloordiv.getKind() == mlir::AffineExprKind::FloorDiv)) {

            if (!exprContainsDim(lhsFloordiv.getLHS(), dimPos)) {
              continue;
            }
            foundNestedFloordiv = true;

            mlir::AffineExpr innerRHS = lhsFloordiv.getRHS();
            if (auto innerDivisor =
                    llvm::dyn_cast<mlir::AffineConstantExpr>(innerRHS)) {
              mlir::AffineExpr combinedDivisor = mlir::getAffineConstantExpr(
                  divisor * innerDivisor.getValue(), expr.getContext());

              mlir::AffineExpr fusedExpr =
                  lhsFloordiv.getLHS().floorDiv(combinedDivisor);
              fusedDivResult = analyzeGridResultExprForDiscontinuity(
                  fusedExpr, dimBounds, dimPos);
            }
          } else if (exprContainsDim(sumOperand, dimPos)) {
            foundNonDivExprWithTargetDim = true;
          }
        }
        if (foundNestedFloordiv && !foundNonDivExprWithTargetDim) {
          return fusedDivResult;
        }

        for (auto sumOperand : sumOperands) {
          // Case 0: constants - accumulate into constSum
          if (auto constExpr =
                  llvm::dyn_cast<mlir::AffineConstantExpr>(sumOperand)) {
            constSum += constExpr.getValue();
            continue;
          }

          // Case 1: Plain dim expression (multiplier = 1)
          if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(sumOperand)) {
            if (dimExpr.getPosition() == dimPos) {
              if (foundDim) {
                return 1; // Multiple occurrences of target dim, fallback
              }
              foundDim = true;
              targetDimExpr = sumOperand;
              multiplier = 1;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), 1});
            }
            continue;
          }

          // Case 2: Mul expression (dim * const)
          if (auto mulOp = isBinaryMul(sumOperand)) {
            auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(mulOp->getLHS());
            auto constExpr =
                llvm::dyn_cast<mlir::AffineConstantExpr>(mulOp->getRHS());

            if (!dimExpr || !constExpr) {
              return 1; // Expected dim * const pattern, fallback
            }

            int64_t mulValue = constExpr.getValue();

            if (dimExpr.getPosition() == dimPos) {
              if (foundDim) {
                return 1; // Multiple occurrences of target dim, fallback
              }
              foundDim = true;
              targetDimExpr = sumOperand;
              multiplier = mulValue;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), mulValue});
            }
            continue;
          }

          // Anything else: fallback
          return 1;
        }

        // If we found the target dim, compute worst-case discontinuity
        if (foundDim && multiplier > 0) {
          if (constSum < 0) {
            return 1; // Negative constants not analyzed, fallback
          }

          int64_t gap;
          if (otherDimInfo.empty()) {
            // Simple case: just the target dim and a constant offset
            // Gap is distance from constSum to the next divisor boundary
            int64_t remainder = constSum % divisor;
            gap = (remainder == 0) ? divisor : (divisor - remainder);
          } else {
            // Build multipliers and bounds arrays from tracked dimension info
            llvm::SmallVector<int64_t> otherMultipliers;
            llvm::SmallVector<int64_t> otherBounds;
            for (auto [dimPosition, mul] : otherDimInfo) {
              otherMultipliers.push_back(mul);
              // Bound is (dimSize - 1) since dim ranges from 0 to dimSize-1
              otherBounds.push_back(dimBounds.lookup(dimPosition) - 1);
            }

            // Use minimizeGap to find the best alignment (minimum gap)
            gap = minimizeGap(divisor, otherMultipliers, otherBounds, constSum);
          }

          int64_t stepsNeeded =
              (gap + multiplier - 1) / multiplier; // Ceiling division
          auto it = checkBoundsAndReturn(stepsNeeded);
          return it;
        }
      }
    }

    // Cannot analyze, conservative fallback
    return 1;
  }

  case mlir::AffineExprKind::Add: {
    int64_t lhsResult =
        analyzeGridResultExprForDiscontinuity(lhs, dimBounds, dimPos);
    int64_t rhsResult =
        analyzeGridResultExprForDiscontinuity(rhs, dimBounds, dimPos);

    // Combine results: take the minimum of constrained values
    // -1 means unconstrained, so we filter those out
    if (lhsResult == -1 && rhsResult == -1) {
      return -1; // Both unconstrained
    }
    if (lhsResult == -1) {
      return rhsResult;
    }
    if (rhsResult == -1) {
      return lhsResult;
    }
    return std::gcd(lhsResult, rhsResult);
  }

  case mlir::AffineExprKind::Mod:
  case mlir::AffineExprKind::Mul: {
    // If a mul expression contains the target dim, it implies any change in the
    // dim will change the output unless this op is the LHS of a floorDiv or
    // ceilDiv expression that contains the target dim.
    // if not, by definition it's unconstrained.
    if (!exprContainsDim(lhs, dimPos)) {
      return -1;
    } else {
      return analyzeGridResultExprForDiscontinuity(lhs, dimBounds, dimPos);
    }
  }

  default:
    return 1; // Cannot analyze, conservative fallback
  }
}

/// Analyzes contiguity for a single shard dimension in an affine map.
/// A shard dimension is any dimension with position >= numGridDims.
///
/// @param map The affine map to analyze
/// @param shape The shape providing bounds for each dimension
/// @param numGridDims The number of grid dimensions (shard dims start after)
/// @param shardDimIdx The index of the shard dimension to analyze (0-based)
/// @return Contiguity bound for the dimension:
///         - 1: Error condition or no coalescing possible
///         - Positive value: Maximum coalescable run length for this dim
inline int64_t analyzeSingleShardDimContiguity(mlir::AffineMap map,
                                               mlir::ArrayRef<int64_t> shape,
                                               unsigned numGridDims,
                                               unsigned numGridResults,
                                               unsigned shardDimIdx) {
  unsigned dimPos = numGridDims + shardDimIdx;

  // Phase I: Set all other dims to worst-case constant values.
  // The dim bounds are derived from the shape at each position.
  llvm::DenseMap<int, int64_t> dimBounds;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dimBounds[static_cast<int>(i)] = shape[i];
  }

  // Phase II: Analyze each result expression for contiguity
  int64_t dimContiguity = -1; // Start unconstrained

  for (auto [i, resultExpr] : llvm::enumerate(map.getResults())) {
    int64_t exprBound;
    if (i >= numGridResults) {
      exprBound =
          analyzeShardResultExprForContiguity(resultExpr, dimBounds, dimPos);
    } else {
      exprBound =
          analyzeGridResultExprForDiscontinuity(resultExpr, dimBounds, dimPos);
    }

    // Combine bounds using GCD
    if (exprBound != -1) {
      if (dimContiguity == -1) {
        dimContiguity = exprBound;
      } else {
        dimContiguity = std::gcd(dimContiguity, exprBound);
      }
    }
  }

  // Replace -1 (unconstrained) with the actual shape extent for this dim
  if (dimContiguity == -1) {
    dimContiguity = shape[dimPos];
  }

  return dimContiguity;
}

/// Computes a combined coalescing factor by analyzing all shard dimensions
/// from most minor to least minor. A shard dimension is any dimension with
/// position >= numGridDims.
///
/// The algorithm:
/// 1. First checks stride alignment - for contiguous access, strides must be:
///    - Innermost shard dim has stride 1
///    - Each outer dim has stride = inner_stride * inner_bound
/// 2. Then iterates shard dims from most minor (last) to least minor (first)
/// 3. For each dim, checks if it's constrained (dimContiguity < shape extent)
/// 4. If constrained, this is the largest contiguous dim - computes the
///    coalescing factor by multiplying the dim's contiguity with the size of
///    all more minor dims that are fully contiguous
/// 5. Returns the minimum of stride-based and contiguity-based constraints
///
/// @param map The affine map to analyze
/// @param shape The shape providing bounds for each dimension
/// @param numGridDims The number of grid dimensions (shard dims start after)
/// @return Combined coalescing factor:
///         - std::nullopt: Error condition (non-analyzable expression)
///         - Positive value: Maximum coalescable run length across all shards
inline int64_t analyzeShardDimContiguity(mlir::AffineMap map,
                                         mlir::ArrayRef<int64_t> shape,
                                         unsigned numGridDims,
                                         unsigned numGridResults,
                                         int64_t elemSizeBytes) {
  TT_assert(elemSizeBytes > 0);
  TT_assertv(shape.size() == map.getNumDims(),
             "Shape size must match number of map dimensions");
  TT_assertv(shape.size() % 2 == 0u, "Shape rank must be even");

  unsigned numShardDims = shape.size() - numGridDims;

  if (numShardDims == 0) {
    return 1; // No shard dims means trivially contiguous
  }

  // Do initial pass of simplification; all remaining mod/floordiv expressions
  // actually do something.
  mlir::AffineMap simplifiedMap =
      simplifyAffineMapWithRangeAnalysis(simplifyZeroFloorDiv(map), shape);

  llvm::dbgs() << "[analyzeShardDimContiguity] simplified map: "
               << simplifiedMap << "\n";

  // Analyze each shard dim and store results
  llvm::SmallVector<int64_t> dimContiguities;
  dimContiguities.reserve(numShardDims);

  for (unsigned shardDimIdx = 0; shardDimIdx < numShardDims; ++shardDimIdx) {
    int64_t contiguity = analyzeSingleShardDimContiguity(
        simplifiedMap, shape, numGridDims, numGridResults, shardDimIdx);
    dimContiguities.push_back(contiguity);
  }

  // Iterate from most minor to least minor shard dim.
  // Accumulate the product of fully contiguous minor dims.
  int64_t coalescingFactor = 1;
  for (int dimIdx = numShardDims - 1; dimIdx >= 0; --dimIdx) {
    unsigned dimPos = numGridDims + dimIdx;
    int64_t dimExtent = shape[dimPos];
    int64_t dimContiguity = dimContiguities[dimIdx];

    coalescingFactor *= dimContiguity;
    if (dimContiguity < dimExtent) {
      // Stop accumulating the coalescing factor once non-contiguous dim is
      // encountered.
      break;
    }
  }
  llvm::dbgs() << "[analyzeShardDimContiguity] coalescingFactor before stride "
                  "analysis = "
               << coalescingFactor << "\n";

  // Check stride alignment. This fundamentally limits the coalescing factor.
  auto strideLimit =
      analyzeShardDimStrides(simplifiedMap, shape, numGridDims, numGridResults);
  TT_assertv(strideLimit % elemSizeBytes == 0,
             "strideLimit must be divisible by elemSizeBytes");
  strideLimit /= elemSizeBytes;

  llvm::dbgs() << "[analyzeShardDimContiguity] strideLimit = " << strideLimit
               << "\n";

  if (strideLimit < coalescingFactor) {
    return std::gcd(coalescingFactor, strideLimit);
  }

  return coalescingFactor;
}

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPUTILS_H
