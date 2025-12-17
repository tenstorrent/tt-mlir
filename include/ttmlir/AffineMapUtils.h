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
      llvm::dbgs()
          << "   [determineWorstCaseLHSValue] use lcm to find max: modulus="
          << modRHSValue << ", multiplier=" << multiplier
          << ", maxValue: " << maxValue << "\n";
    } else {
      maxValue = dimMax * multiplier;
    }
    return {dimOp->getPosition(), mlir::getAffineConstantExpr(maxValue, ctx)};
  }
  return {-1, expr};
}

// WARNING: This function DOES NOT produce usable affine maps; it is _strictly_
// for analyzing what-if scenarios where some dims are held constant.
//
// This function simplifies expressions of the form: (dim * A + dim * B ... )
// mod C, which can help prove that dim subsets (i.e. shard dims) are pure
// affine expressions if other dims are held constant.
//
// constDimBounds maps dimension positions to their bounds (number of elements).
// For a dim with bound B, the max value of dim is B-1.
inline mlir::AffineExpr replaceDimsWithWorstCaseValues(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &constDimBounds) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = replaceDimsWithWorstCaseValues(binOp.getLHS(), constDimBounds);
    auto rhs = replaceDimsWithWorstCaseValues(binOp.getRHS(), constDimBounds);

    bool isMod = binOp.getKind() == mlir::AffineExprKind::Mod;
    bool isFloorDiv = binOp.getKind() == mlir::AffineExprKind::FloorDiv;
    if (isMod || isFloorDiv) {
      if (auto modulus = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        llvm::dbgs() << "   [replaceDimsWithWorstCaseValues] expr: " << expr
                     << "\n";
        // detect pattern (dim * A + dim * B ... ) mod C
        if (auto sumOperands = collectSumOperands(lhs); !sumOperands.empty()) {
          llvm::SmallVector<std::pair<unsigned, mlir::AffineExpr>>
              newSumOperands;
          for (auto sumOperand : sumOperands) {
            newSumOperands.push_back(determineWorstCaseLHSValue(
                sumOperand, constDimBounds, modulus, expr.getContext()));
          }
          llvm::dbgs()
              << "   [replaceDimsWithWorstCaseValues] newSumOperands: \n";
          for (auto newSumOperand : newSumOperands) {
            llvm::dbgs()
                << "     [replaceDimsWithWorstCaseValues] newSumOperand: "
                << newSumOperand.second << "\n";
          }

          // Categorize sum operands into constant and non-constant expressions.
          int64_t combinedGcd = 0;
          llvm::SmallVector<std::pair<unsigned, mlir::AffineConstantExpr>>
              constExprs;
          llvm::SmallVector<mlir::AffineExpr> nonConstExprs;
          for (auto [dimPos, sumExpr] : newSumOperands) {
            if (auto constExpr =
                    llvm::dyn_cast<mlir::AffineConstantExpr>(sumExpr)) {
              constExprs.push_back({dimPos, constExpr});
            } else {
              nonConstExprs.push_back(sumExpr);
            }
          }

          mlir::AffineExpr newSumExpr =
              mlir::getAffineConstantExpr(0, expr.getContext());

          // Find holistic worst-case by combining constant expressions to
          // find worst-case alignment relative to the modulus.
          for (auto [dimPos, constExpr] : constExprs) {
            auto constVal = constExpr.getValue();
            if (modulus.getValue() > constVal) {
              int64_t diff = modulus.getValue() - constExpr.getValue();
              if (diff > 0) {
                combinedGcd =
                    (combinedGcd == 0) ? diff : std::gcd(combinedGcd, diff);
              }
            }
          }
          int64_t combinedConst =
              (combinedGcd > 0) ? (modulus.getValue() - combinedGcd) : 0;

          // find raw sum of all constExpr
          int64_t constSum = 0;
          for (auto [dimPos, constExpr] : constExprs) {
            constSum += constExpr.getValue();
          }

          // use raw sum if GCD exceeds max sum of all constExprs
          if (combinedConst <= constSum) {
            newSumExpr =
                mlir::getAffineConstantExpr(combinedConst, expr.getContext());
          } else {
            newSumExpr =
                mlir::getAffineConstantExpr(constSum, expr.getContext());
          }

          // Add all non-constant expressions to the result
          for (auto nonConstExpr : nonConstExprs) {
            newSumExpr = newSumExpr + nonConstExpr;
          }
          lhs = newSumExpr;
        }
      }
      return (isMod) ? (lhs % rhs) : (lhs.floorDiv(rhs));
    }

    switch (binOp.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      return expr;
    }
  }
  return expr;
}

inline mlir::AffineMap replaceDimExprsWithConstValues(
    mlir::AffineMap map, const llvm::DenseMap<int, int64_t> &constDimBounds) {
  mlir::SmallVector<mlir::AffineExpr> newResults;
  for (auto result : map.getResults()) {
    auto simplified = replaceDimsWithWorstCaseValues(result, constDimBounds);
    newResults.push_back(simplified);
  }
  return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                              map.getContext());
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

/// Helper function to recursively analyze contiguity of an affine expression.
/// This identifies the maximum coalescable run by analyzing mod/floordiv
/// expressions.
///
/// @param expr The expression to analyze
/// @param rhsConst Optional bound from enclosing mod/floordiv operation.
///                 When set, changes how expressions are evaluated.
/// @return std::nullopt for error condition,
///         -1 for unconstrained (no contiguity limit),
///         positive value for the contiguity bound
inline std::optional<int64_t>
analyzeContiguityExpr(mlir::AffineExpr expr,
                      std::optional<int64_t> parentModulus = std::nullopt) {
  // Handle constant expressions
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    if (constExpr.getValue() < 0) {
      return std::nullopt; // Error: negative constant
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
    return std::nullopt;
  }

  mlir::AffineExpr lhs = binOp.getLHS();
  mlir::AffineExpr rhs = binOp.getRHS();

  switch (binOp.getKind()) {
  case mlir::AffineExprKind::Add: {
    // Analyze each operand recursively, propagating rhsConst
    auto lhsBound = analyzeContiguityExpr(lhs, parentModulus);
    auto rhsBound = analyzeContiguityExpr(rhs, parentModulus);

    // If either operand returned nullopt, return nullopt
    if (!lhsBound.has_value() || !rhsBound.has_value()) {
      return std::nullopt;
    }

    // Return GCD of bounds; -1 means unconstrained
    if (*lhsBound == -1) {
      return *rhsBound;
    }
    if (*rhsBound == -1) {
      return *lhsBound;
    }
    return std::gcd(*lhsBound, *rhsBound);
  }

  case mlir::AffineExprKind::Mul: {
    // Mul must have a constant operand
    auto lhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs);
    auto rhsConstExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);

    if (!lhsConst && !rhsConstExpr) {
      return std::nullopt; // Error: mul without constant operand
    }

    int64_t constVal =
        rhsConstExpr ? rhsConstExpr.getValue() : lhsConst.getValue();

    if (constVal < 0) {
      return std::nullopt; // Error: negative constant
    }

    auto lhsBound = analyzeContiguityExpr(lhs, parentModulus);
    // if lhs cannot be analyzed, return error
    if (!lhsBound) {
      return std::nullopt;
    }

    // if lhs is constrained, propagate the constraint up
    if (lhsBound.value() != -1) {
      return lhsBound.value();
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
    return std::nullopt;
  }

  case mlir::AffineExprKind::Mod:
  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    // RHS must be constant
    auto rhsConstExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    assert(rhsConstExpr && "Mod/FloorDiv RHS must be constant expression");

    int64_t modulus = rhsConstExpr.getValue();
    if (modulus <= 0) {
      return std::nullopt;
    }

    // Check if LHS is also floordiv/mod - error condition
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs)) {
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Mod ||
          lhsBinOp.getKind() == mlir::AffineExprKind::FloorDiv) {
        return std::nullopt;
      }
    }

    // If LHS is plain dim expr, return the mod/floordiv value
    if (llvm::isa<mlir::AffineDimExpr>(lhs)) {
      return modulus;
    }

    // If LHS is a Mul, return result from calling func on lhs with boundValue
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs)) {
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Mul) {
        return analyzeContiguityExpr(lhs, modulus);
      }

      // If LHS is an Add, figure out largest value that keeps lhs < boundValue
      // We recursively analyze with boundValue as the constraint
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Add) {
        assert(!llvm::isa<mlir::AffineConstantExpr>(lhsBinOp.getLHS()));

        // get a const multiplier on the LHS, if it exists
        int multiplier = 1;
        if (auto mulOp = isBinaryMul(lhsBinOp.getLHS())) {
          if (auto multiplierExpr =
                  mlir::dyn_cast<mlir::AffineConstantExpr>(mulOp->getRHS())) {
            multiplier = multiplierExpr.getValue();
          } else {
            return std::nullopt;
          }
        }

        if (auto addRHS =
                llvm::dyn_cast<mlir::AffineConstantExpr>(lhsBinOp.getRHS())) {
          auto modRHSConst = addRHS.getValue() % modulus;
          auto gap = modulus - modRHSConst;
          return (gap + multiplier - 1) / multiplier;
        }
        return std::nullopt;
      }
    }

    // Default: recursively analyze LHS with boundValue as the new constraint
    return analyzeContiguityExpr(lhs, modulus);
  }

  default:
    return std::nullopt;
  }
}

/// Analyzes contiguity for a single shard dimension in an affine map.
/// A shard dimension is any dimension with position >= numGridDims.
///
/// For the specified shard dimension, this function:
/// 1. Phase I: Isolates the dimension by setting all other dims to worst-case
///    values using replaceDimsWithWorstCaseValues(). The dim bounds are derived
///    from the shape at each position.
/// 2. Phase II: Recursively analyzes the resulting expression to find the
///    maximum coalescable run by examining mod/floordiv expressions.
///
/// @param map The affine map to analyze
/// @param shape The shape providing bounds for each dimension
/// @param numGridDims The number of grid dimensions (shard dims start after)
/// @param shardDimIdx The index of the shard dimension to analyze (0-based)
/// @return Optional contiguity bound for the dimension:
///         - std::nullopt: Error condition (non-analyzable expression)
///         - Positive value: Maximum coalescable run length for this dim
inline std::optional<int64_t>
analyzeSingleShardDimContiguity(mlir::AffineMap map,
                                mlir::ArrayRef<int64_t> shape,
                                unsigned numGridDims, unsigned shardDimIdx) {
  unsigned dimPos = numGridDims + shardDimIdx;

  // Phase I: Set all other dims to worst-case constant values.
  // The dim bounds are derived from the shape at each position.
  llvm::DenseMap<int, int64_t> constDimBounds;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    if (i != dimPos) {
      constDimBounds[static_cast<int>(i)] = shape[i];
    }
  }

  // Do initial pass of simplification; all remaining mod/floordiv expressions
  // actually do something.
  llvm::SmallVector<int64_t> dimBounds(shape.begin(), shape.end());
  mlir::AffineMap simplifiedMap =
      simplifyAffineMapWithRangeAnalysis(simplifyZeroFloorDiv(map), dimBounds);

  // Apply worst-case replacement to isolate the target shard dim
  mlir::AffineMap constReplacedMap =
      replaceDimExprsWithConstValues(simplifiedMap, constDimBounds);
  llvm::dbgs() << "[analyzeSingleShardDimContiguity] const replaced map: "
               << constReplacedMap << "\n";

  // Simplify the map again using range analysis
  mlir::AffineMap fullyIsolatedMap =
      simplifyAffineMapWithRangeAnalysis(constReplacedMap, dimBounds);
  llvm::dbgs() << "[analyzeSingleShardDimContiguity] fully isolated map: "
               << fullyIsolatedMap << "\n";

  // Phase II: Analyze each result expression for contiguity
  std::optional<int64_t> dimContiguity = -1; // Start unconstrained

  for (mlir::AffineExpr resultExpr : fullyIsolatedMap.getResults()) {
    auto exprBound = analyzeContiguityExpr(resultExpr);
    llvm::dbgs() << "   [analyzeSingleShardDimContiguity] for result: "
                 << resultExpr
                 << " --> expr bound: " << exprBound.value_or(-777) << "\n";

    if (!exprBound.has_value()) {
      return std::nullopt;
    }

    // Combine bounds using GCD
    if (*exprBound != -1) {
      if (*dimContiguity == -1) {
        dimContiguity = *exprBound;
      } else {
        dimContiguity = std::gcd(*dimContiguity, *exprBound);
      }
    }
  }

  // Replace -1 (unconstrained) with the actual shape extent for this dim
  if (dimContiguity.has_value() && *dimContiguity == -1) {
    dimContiguity = shape[dimPos];
  }

  return dimContiguity;
}

/// Computes a combined coalescing factor by analyzing all shard dimensions
/// from most minor to least minor. A shard dimension is any dimension with
/// position >= numGridDims.
///
/// The algorithm:
/// 1. Iterates shard dims from most minor (last) to least minor (first)
/// 2. For each dim, checks if it's constrained (dimContiguity < shape extent)
/// 3. If constrained, this is the largest contiguous dim - computes the
///    coalescing factor by multiplying the dim's contiguity with the size of
///    all more minor dims that are fully contiguous
/// 4. If all dims are fully contiguous, returns the product of all shard sizes
///
/// @param map The affine map to analyze
/// @param shape The shape providing bounds for each dimension
/// @param numGridDims The number of grid dimensions (shard dims start after)
/// @return Combined coalescing factor:
///         - std::nullopt: Error condition (non-analyzable expression)
///         - Positive value: Maximum coalescable run length across all shards
inline std::optional<int64_t>
analyzeShardDimContiguity(mlir::AffineMap map, mlir::ArrayRef<int64_t> shape,
                          unsigned numGridDims) {
  TT_assertv(shape.size() == map.getNumDims(),
             "Shape size must match number of map dimensions");
  TT_assertv(shape.size() % 2 == 0u, "Shape rank must be even");

  unsigned numShardDims = shape.size() - numGridDims;

  if (numShardDims == 0) {
    return 1; // No shard dims means trivially contiguous
  }

  // Analyze each shard dim and store results
  llvm::SmallVector<std::optional<int64_t>> dimContiguities;
  dimContiguities.reserve(numShardDims);

  for (unsigned shardDimIdx = 0; shardDimIdx < numShardDims; ++shardDimIdx) {
    auto contiguity =
        analyzeSingleShardDimContiguity(map, shape, numGridDims, shardDimIdx);
    if (!contiguity.has_value()) {
      return std::nullopt; // Propagate error
    }
    dimContiguities.push_back(contiguity);
  }

  // Iterate from most minor to least minor shard dim
  // Accumulate the product of fully contiguous minor dims
  int64_t minorProduct = 1;

  for (int dimIdx = numShardDims - 1; dimIdx >= 0; --dimIdx) {
    unsigned dimPos = numGridDims + dimIdx;
    int64_t dimExtent = shape[dimPos];
    int64_t dimContiguity = *dimContiguities[dimIdx];

    if (dimContiguity < dimExtent) {
      // This dim is constrained - it's the largest contiguous dim
      // Return dimContiguity * product of all fully contiguous minor dims
      return dimContiguity * minorProduct;
    }

    // Dim is fully contiguous, accumulate its size
    minorProduct *= dimExtent;
  }

  // All dims are fully contiguous
  return minorProduct;
}

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPUTILS_H
