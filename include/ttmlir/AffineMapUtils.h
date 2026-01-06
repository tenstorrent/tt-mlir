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
#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

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

  // Collect all grid indices first
  std::vector<llvm::SmallVector<int64_t>> gridIndices;
  sample(gridShape, [&](mlir::ArrayRef<int64_t> gridIndex) {
    gridIndices.push_back(
        llvm::SmallVector<int64_t>(gridIndex.begin(), gridIndex.end()));
  });

  // If no grid indices, return shard volume
  if (gridIndices.empty()) {
    return shardVolume;
  }

  // Determine number of threads to use
  unsigned numThreads = std::min(static_cast<unsigned>(8),
                                 static_cast<unsigned>(gridIndices.size()));
  if (numThreads == 0) {
    numThreads = 1;
  }

  // Use atomic counter for dynamic work assignment
  std::atomic<size_t> nextIndex{0};
  std::vector<size_t> threadResults(numThreads, shardVolume);
  std::vector<std::thread> threads;

  // Thread pool: each thread dynamically pulls work items
  for (unsigned threadId = 0; threadId < numThreads; ++threadId) {
    threads.emplace_back([&, threadId]() {
      size_t localCoalescingFactor = shardVolume;
      mlir::SmallVector<int64_t> memoryIndex;
      memoryIndex.resize(gridShape.size() + shardShape.size());

      // Dynamically pull work items until all are processed
      while (true) {
        // Atomically fetch and increment the next index to process
        size_t i = nextIndex.fetch_add(1, std::memory_order_relaxed);

        // Check if all work is done
        if (i >= gridIndices.size()) {
          break;
        }

        // Early exit optimization: if coalescing factor is already 1, stop
        if (localCoalescingFactor == 1) {
          break;
        }

        const auto &gridIndex = gridIndices[i];
        size_t currentCoalescingFactor = 0;
        mlir::SmallVector<int64_t, 4> nextAddress;

        sample(shardShape, [&](mlir::ArrayRef<int64_t> shardIndex) {
          if (localCoalescingFactor == 1) {
            return;
          }

          // Build memory index
          for (unsigned j = 0; j < gridIndex.size(); j++) {
            memoryIndex[j] = gridIndex[j];
          }
          for (unsigned j = 0; j < shardIndex.size(); j++) {
            memoryIndex[gridIndex.size() + j] = shardIndex[j];
          }

          mlir::SmallVector<int64_t, 4> address = map.compose(memoryIndex);
          if (nextAddress.empty() || nextAddress == address) {
            ++currentCoalescingFactor;
          } else {
            localCoalescingFactor =
                std::gcd(localCoalescingFactor, currentCoalescingFactor);
            if (localCoalescingFactor == 1) {
              return;
            }
            currentCoalescingFactor = 1;
          }
          nextAddress = address;
          nextAddress.back() += stride;
        });

        // Account for final run
        localCoalescingFactor =
            std::gcd(localCoalescingFactor, currentCoalescingFactor);
      }

      threadResults[threadId] = localCoalescingFactor;
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Merge results from all threads using GCD
  size_t coalescingFactor = threadResults[0];
  for (size_t i = 1; i < threadResults.size(); ++i) {
    coalescingFactor = std::gcd(coalescingFactor, threadResults[i]);
    if (coalescingFactor == 1) {
      break;
    }
  }

  return coalescingFactor;
}

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPUTILS_H
