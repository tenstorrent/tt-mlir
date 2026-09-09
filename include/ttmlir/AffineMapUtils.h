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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"

#include <numeric>

namespace ttmlir::utils {

/// Returns a new shape by applying `map` to the input shape.
template <typename Vector>
llvm::SmallVector<int64_t> evalShape(mlir::AffineMap map, Vector shape) {
  // End point of the input box: (shape[0]-1, shape[1]-1, ...).
  llvm::SmallVector<int64_t, 4> endCoord;
  for (auto dim : shape) {
    endCoord.push_back(dim - 1);
  }
  // Start (0,0,...); the other corner used for the extent formula.
  llvm::SmallVector<int64_t, 4> startCoord(endCoord.size(), 0);

  auto mappedEndCoord = map.compose(endCoord);
  auto mappedStartCoord = map.compose(startCoord);

  // Size in each dimension = (mapped end) - (mapped start) + 1.
  llvm::SmallVector<int64_t, 4> result;
  for (size_t i = 0; i < mappedEndCoord.size(); ++i) {
    result.push_back(mappedEndCoord[i] - mappedStartCoord[i] + 1);
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

/// Returns a new affine map with constant offsets applied to a contiguous
/// subset of map results.
inline mlir::AffineMap applyOffsetsToAffineMapResults(
    mlir::AffineMap map, mlir::ArrayRef<int64_t> offsets, unsigned startIndex) {
  if (offsets.empty()) {
    return map;
  }
  mlir::MLIRContext *ctx = map.getContext();
  TT_assertv(startIndex + offsets.size() <= map.getNumResults(),
             "Result offset range out of bounds");
  unsigned endIndex = startIndex + offsets.size();
  llvm::SmallVector<mlir::AffineExpr> remappedResults(map.getResults().begin(),
                                                      map.getResults().end());
  for (unsigned i = startIndex; i < endIndex; ++i) {
    remappedResults[i] = remappedResults[i] +
                         getAffineConstantExpr(offsets[i - startIndex], ctx);
  }
  return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                              remappedResults, ctx);
}

/// Returns a new affine map with constant offsets applied to a contiguous
/// subset of map input dimensions by replacing dim uses with (dim + offset).
inline mlir::AffineMap applyOffsetsToAffineMapDims(
    mlir::AffineMap map, mlir::ArrayRef<int64_t> offsets, unsigned startIndex) {
  if (offsets.empty()) {
    return map;
  }
  mlir::MLIRContext *ctx = map.getContext();
  TT_assertv(startIndex + offsets.size() <= map.getNumDims(),
             "Dim offset range out of bounds");
  unsigned endIndex = startIndex + offsets.size();
  llvm::SmallVector<mlir::AffineExpr> dimReplacements;
  dimReplacements.reserve(map.getNumDims());
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dimReplacements.push_back(getAffineDimExpr(i, ctx));
  }
  for (unsigned i = startIndex; i < endIndex; ++i) {
    dimReplacements[i] = dimReplacements[i] +
                         getAffineConstantExpr(offsets[i - startIndex], ctx);
  }
  return map.replaceDimsAndSymbols(dimReplacements, {}, map.getNumDims(),
                                   map.getNumSymbols());
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

/// Returns a new affine map by dropping the specified dimension.
// E.g.
// input: (d0, d1, d2) -> (0, d1, d2), dimToRemove=0,
//  then the output is: (d0, d1) -> (0, d0, d1)
// input: (d0, d1, d2) -> (0, d1, d2), dimToRemove=1,
//  then the output is: (d0, d1) -> (0, 0, d1)
inline mlir::AffineMap dropDim(mlir::AffineMap map, unsigned dimToRemove) {
  mlir::MLIRContext *ctx = map.getContext();
  unsigned numDims = map.getNumDims();
  assert(dimToRemove < numDims && "dim out of range");

  llvm::SmallVector<mlir::AffineExpr> replacements;
  for (unsigned i = 0; i < numDims; ++i) {
    if (i < dimToRemove) {
      // Dims before removed one stay the same.
      replacements.push_back(getAffineDimExpr(i, ctx));
    } else if (i == dimToRemove) {
      // Replace removed dim with constant 0 (or could be any value).
      replacements.push_back(getAffineConstantExpr(0, ctx));
    } else {
      // Dims after shift down by 1.
      replacements.push_back(getAffineDimExpr(i - 1, ctx));
    }
  }

  return map.replaceDimsAndSymbols(replacements, {}, numDims - 1,
                                   map.getNumSymbols());
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

// Utility function to create an identity forward map for grid virtualization
// Returns a map: (d0, d1) -> (0, d0, d1)
inline mlir::AffineMap
createIdentityGridForwardMap(mlir::MLIRContext *context) {
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, context);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, context);
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, context);
  return mlir::AffineMap::get(2, 0, {zero, d0, d1}, context);
}

// Derives a grid inverse map _specifically_ for 2D->2D permutation index maps
// that fit inside the target grid in both dimensions. In those cases, instead
// of doing a standard reblocking (which behaves like a reshape), we can instead
// simply _permute_ grid indices for virtual grid forward and inverse mappings.
//
// Asserts if there is no inverse permutation possible.
// If the index map provided is empty or identity, returns an identity grid
// inverse map.
inline std::pair<mlir::AffineMap, mlir::AffineMap>
createGridForwardAndInverseMapFor2DPermutation(mlir::AffineMap indexMap,
                                               unsigned gridRank,
                                               mlir::MLIRContext *context) {
  // If no index_map or it's empty/identity, return identity grid inverse map
  if (!indexMap || indexMap.isEmpty() || indexMap.isIdentity()) {
    return {createIdentityGridForwardMap(context),
            createIdentityGridInverseMap(context)};
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
  TT_assertv(invGridMap,
             "Index map is not a valid permutation for grid inverse");

  // Build grid inverse map with device ID prefix: (d0, d1) -> (0, inv_y, inv_x)
  mlir::AffineExpr zero = mlir::getAffineConstantExpr(0, context);
  llvm::SmallVector<mlir::AffineExpr> invResults;
  invResults.push_back(zero);
  for (auto result : invGridMap.getResults()) {
    invResults.push_back(result);
  }
  auto invMap = mlir::AffineMap::get(gridRank, 0, invResults, context);
  auto fwdMap = gridMap.insertResult(zero, 0);

  return {fwdMap, invMap};
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

/// Calculate the new tensor shape when reblocking to a new grid shape.
/// This is the shape-only variant of calculateReblockMapForGrid.
inline mlir::SmallVector<int64_t>
calculateReblockShapeForGrid(mlir::ArrayRef<int64_t> tensorShape,
                             mlir::ArrayRef<int64_t> newGridShape) {
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
  return newTensorShape;
}

/// Calculate a reblock affine map given a shape and new grid shape.
/// Returns the new tensor shape and the reblock affine map.
inline std::pair<mlir::SmallVector<int64_t>, mlir::AffineMap>
calculateReblockMapForGrid(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::ArrayRef<int64_t> newGridShape,
                           mlir::MLIRContext *context) {
  auto newTensorShape = calculateReblockShapeForGrid(tensorShape, newGridShape);
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

/// Creates an affine map that collapses an ND grid (> 2D) to a 2D grid while
/// preserving shard dimensions as identity pass-throughs.
///
/// The map linearizes ND grid coordinates to a 1D row-major index, then
/// re-expands to the target 2D physical grid. Shard dimensions are appended
/// as identity.
///
/// Example: gridShape=[7,1,1], physGrid=[7,1], shardRank=3:
///   (d0,d1,d2,d3,d4,d5) -> ((d0+d1+d2)%7, 0, d3, d4, d5)
inline mlir::AffineMap
createNDGridCollapseMap(mlir::ArrayRef<int64_t> gridShape,
                        mlir::ArrayRef<int64_t> physGrid, unsigned shardRank,
                        mlir::MLIRContext *context) {
  assert(physGrid.size() == 2 && "Physical grid must be 2D");
  unsigned gridRank = gridShape.size();

  // Linearize ND grid → 1D (row-major).
  mlir::AffineExpr linearExpr = mlir::getAffineConstantExpr(0, context);
  mlir::AffineExpr stride = mlir::getAffineConstantExpr(1, context);
  for (int64_t i = static_cast<int64_t>(gridRank) - 1; i >= 0; --i) {
    linearExpr = linearExpr + mlir::getAffineDimExpr(i, context) * stride;
    stride = stride * mlir::getAffineConstantExpr(gridShape[i], context);
  }

  // Expand 1D → 2D physical grid.
  mlir::SmallVector<mlir::AffineExpr> results;
  mlir::AffineExpr divisor = mlir::getAffineConstantExpr(1, context);
  for (int64_t dim = 1; dim >= 0; --dim) {
    mlir::AffineExpr sizeExpr =
        mlir::getAffineConstantExpr(physGrid[dim], context);
    results.insert(results.begin(), linearExpr.floorDiv(divisor) % sizeExpr);
    divisor = divisor * sizeExpr;
  }

  // Append identity pass-throughs for shard dims.
  for (unsigned i = 0; i < shardRank; ++i) {
    results.push_back(mlir::getAffineDimExpr(gridRank + i, context));
  }

  return mlir::AffineMap::get(gridRank + shardRank, 0, results, context);
}

namespace detail {

// A self-contained drop-in replacement of AffineMap::compose(ArrayRef<int64_t>)
// that compiles & evaluates affine maps using a register-based bytecode VM.
class CompiledAffineMap {
public:
  explicit CompiledAffineMap(mlir::AffineMap map)
      : numResults_(map.getNumResults()) {
    llvm::DenseMap<mlir::AffineExpr, uint16_t> exprToReg;
    resultRegs_.reserve(numResults_);
    for (int i = 0; i < numResults_; ++i) {
      resultRegs_.push_back(compileExpr(map.getResult(i), exprToReg));
    }
  }

  void evaluate(const int64_t *__restrict inputs,
                int64_t *__restrict results) const {
    int64_t regs[kMaxRegs];
    const Instruction *code = code_.data();
    unsigned codeSize = static_cast<unsigned>(code_.size());
    for (unsigned pc = 0; pc < codeSize; ++pc) {
      int64_t a, b, q, rem;
      switch (code[pc].op) {
      case Op::LoadDim:
        regs[code[pc].dst] = inputs[code[pc].imm];
        break;
      case Op::LoadConst:
        regs[code[pc].dst] = code[pc].imm;
        break;
      case Op::Add:
        regs[code[pc].dst] = regs[code[pc].src1] + regs[code[pc].src2];
        break;
      case Op::Mul:
        regs[code[pc].dst] = regs[code[pc].src1] * regs[code[pc].src2];
        break;
      case Op::FloorDiv:
        a = regs[code[pc].src1];
        b = regs[code[pc].src2];
        q = a / b;
        rem = a % b;
        regs[code[pc].dst] = (rem != 0 && ((rem ^ b) < 0)) ? q - 1 : q;
        break;
      case Op::Mod:
        a = regs[code[pc].src1];
        b = regs[code[pc].src2];
        rem = a % b;
        regs[code[pc].dst] = (rem != 0 && ((rem ^ b) < 0)) ? rem + b : rem;
        break;
      case Op::CeilDiv:
        a = regs[code[pc].src1];
        b = regs[code[pc].src2];
        q = a / b;
        rem = a % b;
        regs[code[pc].dst] = (rem != 0 && ((rem ^ b) > 0)) ? q + 1 : q;
        break;
      case Op::ShrImm:
        regs[code[pc].dst] = regs[code[pc].src1] >> code[pc].imm;
        break;
      case Op::AndImm:
        regs[code[pc].dst] = regs[code[pc].src1] & code[pc].imm;
        break;
      }
    }
    for (int i = 0; i < numResults_; ++i) {
      results[i] = regs[resultRegs_[i]];
    }
  }

  int getNumResults() const { return numResults_; }

private:
  static constexpr uint16_t kMaxRegs = 512;

  enum class Op : uint8_t {
    LoadDim,
    LoadConst,
    Add,
    Mul,
    FloorDiv,
    Mod,
    CeilDiv,
    ShrImm,
    AndImm,
  };

  struct Instruction {
    Op op;
    uint16_t dst;
    uint16_t src1;
    uint16_t src2;
    int64_t imm;
  };

  static bool isPowerOf2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

  uint16_t emitInstruction(mlir::AffineExpr expr, Instruction inst,
                           llvm::DenseMap<mlir::AffineExpr, uint16_t> &map) {
    const uint16_t reg = static_cast<uint16_t>(code_.size());
    TT_assert(reg < kMaxRegs);
    inst.dst = reg;
    code_.push_back(inst);
    // Memoization for CSE.
    map[expr] = reg;
    return reg;
  }

  uint16_t compileExpr(mlir::AffineExpr expr,
                       llvm::DenseMap<mlir::AffineExpr, uint16_t> &exprToReg) {
    // CSE: MLIR AffineExpr objects are pointer-uniqued, compile shared
    // subexpressions across results once and reuse.
    auto it = exprToReg.find(expr);
    if (it != exprToReg.end()) {
      return it->second;
    }

    switch (expr.getKind()) {
    case mlir::AffineExprKind::DimId:
      return emitInstruction(
          expr,
          {Op::LoadDim, 0, 0, 0,
           static_cast<int64_t>(
               mlir::cast<mlir::AffineDimExpr>(expr).getPosition())},
          exprToReg);
    case mlir::AffineExprKind::SymbolId:
      llvm_unreachable("Symbols not supported in compiled affine map");
    case mlir::AffineExprKind::Constant:
      return emitInstruction(
          expr,
          {Op::LoadConst, 0, 0, 0,
           mlir::cast<mlir::AffineConstantExpr>(expr).getValue()},
          exprToReg);
    default:
      break;
    }

    auto bin = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    const uint16_t lhs = compileExpr(bin.getLHS(), exprToReg);

    // Power-of-2 specialization for FloorDiv and Mod: the RHS of these
    // operations in MLIR affine expressions is always a constant.
    if (expr.getKind() == mlir::AffineExprKind::FloorDiv ||
        expr.getKind() == mlir::AffineExprKind::Mod) {
      if (auto constRHS =
              mlir::dyn_cast<mlir::AffineConstantExpr>(bin.getRHS())) {
        int64_t val = constRHS.getValue();
        if (isPowerOf2(val)) {
          if (expr.getKind() == mlir::AffineExprKind::FloorDiv) {
            return emitInstruction(
                expr,
                {Op::ShrImm, 0, lhs, 0,
                 llvm::countr_zero(static_cast<uint64_t>(val))},
                exprToReg);
          }
          return emitInstruction(expr, {Op::AndImm, 0, lhs, 0, val - 1},
                                 exprToReg);
        }
      }
    }

    const uint16_t rhs = compileExpr(bin.getRHS(), exprToReg);

    Op binOp;
    switch (expr.getKind()) {
    case mlir::AffineExprKind::Add:
      binOp = Op::Add;
      break;
    case mlir::AffineExprKind::Mul:
      binOp = Op::Mul;
      break;
    case mlir::AffineExprKind::Mod:
      binOp = Op::Mod;
      break;
    case mlir::AffineExprKind::FloorDiv:
      binOp = Op::FloorDiv;
      break;
    case mlir::AffineExprKind::CeilDiv:
      binOp = Op::CeilDiv;
      break;
    default:
      llvm_unreachable("unexpected affine expr kind");
    }
    return emitInstruction(expr, {binOp, 0, lhs, rhs, 0}, exprToReg);
  }

  llvm::SmallVector<Instruction, 128> code_;
  llvm::SmallVector<uint16_t, 8> resultRegs_;
  int numResults_;
};

} // namespace detail

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
                                        const int64_t stride,
                                        const unsigned numGridDims = 0) {
  TT_assertv(map.getNumDims() == shape.size(),
             "Map dimensions must match shape size");
  TT_assertv(numGridDims <= shape.size(),
             "Number of grid dims cannot exceed shape size");
  TT_assertv(map.getNumSymbols() == 0u,
             "calculateCoalescingFactor expects a symbol-free affine map");

  mlir::ArrayRef<int64_t> gridShape = shape.take_front(numGridDims);
  mlir::ArrayRef<int64_t> shardShape = shape.drop_front(numGridDims);

  // If no shard dims, trivially contiguous (volume is 1).
  if (shardShape.empty()) {
    return 1;
  }

  detail::CompiledAffineMap compiled(map);
  const int numResults = compiled.getNumResults();
  constexpr int kMaxResults = 16;
  constexpr int kMaxDims = 16;
  TT_assertv(numResults <= kMaxResults,
             "calculateCoalescingFactor expects <= 16 map results");
  TT_assertv(static_cast<int>(map.getNumDims()) <= kMaxDims,
             "calculateCoalescingFactor expects <= 16 map dims");

  const int64_t shardVolume = volume(shardShape);
  int64_t coalescingFactor = shardVolume;

  int64_t memoryIndex[kMaxDims] = {-1};

  sample(gridShape, [&](mlir::ArrayRef<int64_t> gridIndex) {
    if (coalescingFactor == 1) {
      return;
    }
    for (unsigned i = 0; i < gridIndex.size(); i++) {
      memoryIndex[i] = gridIndex[i];
    }
    for (unsigned i = 0; i < shardShape.size(); i++) {
      memoryIndex[numGridDims + i] = 0;
    }

    int64_t currentCoalescingFactor = 0;
    int64_t address[kMaxResults] = {-1};
    int64_t nextAddress[kMaxResults] = {-1};

    for (int64_t iter = 0; iter < shardVolume; iter++) {
      compiled.evaluate(memoryIndex, address);

      bool contiguous = false;
      if (iter > 0) {
        contiguous = true;
        for (int i = 0; i < numResults; i++) {
          if (address[i] != nextAddress[i]) {
            contiguous = false;
            break;
          }
        }
      }

      if (iter == 0 || contiguous) {
        ++currentCoalescingFactor;
      } else {
        coalescingFactor = std::gcd(coalescingFactor, currentCoalescingFactor);
        if (coalescingFactor == 1) {
          break;
        }
        // Current memory access can potentially be coalesced with next access!
        currentCoalescingFactor = 1;
      }

      for (int i = 0; i < numResults; i++) {
        nextAddress[i] = address[i];
      }
      nextAddress[numResults - 1] += stride;

      // Increment shard coordinates (assuming row-major order).
      for (int i = static_cast<int>(shardShape.size()) - 1; i >= 0; i--) {
        memoryIndex[numGridDims + i]++;
        if (memoryIndex[numGridDims + i] < shardShape[i]) {
          break;
        }
        memoryIndex[numGridDims + i] = 0;
      }
    }
    // Account for the final run.
    coalescingFactor = std::gcd(coalescingFactor, currentCoalescingFactor);
  });

  return static_cast<size_t>(coalescingFactor);
}

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPUTILS_H
