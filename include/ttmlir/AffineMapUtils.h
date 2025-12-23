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
  size_t coalescingFactor = shardVolume;

  mlir::SmallVector<int64_t> memoryIndex;
  memoryIndex.resize(gridShape.size() + shardShape.size());
  sample(gridShape, [&](mlir::ArrayRef<int64_t> gridIndex) {
    if (coalescingFactor == 1) {
      return;
    }
    size_t currentCoalescingFactor = 0;
    mlir::SmallVector<int64_t, 4> nextAddress;
    sample(shardShape, [&](mlir::ArrayRef<int64_t> shardIndex) {
      if (coalescingFactor == 1) {
        return;
      }
      for (unsigned i = 0; i < gridIndex.size(); i++) {
        memoryIndex[i] = gridIndex[i];
      }
      for (unsigned i = 0; i < shardIndex.size(); i++) {
        memoryIndex[gridIndex.size() + i] = shardIndex[i];
      }
      mlir::SmallVector<int64_t, 4> address = map.compose(memoryIndex);
      if (nextAddress.empty() || nextAddress == address) {
        ++currentCoalescingFactor;
      } else {
        coalescingFactor = std::gcd(coalescingFactor, currentCoalescingFactor);
        // If coalescing factor reaches unit size, it cannot change further.
        // Early exit to save on runtime.
        if (coalescingFactor == 1) {
          return;
        }
        // current memory access can potentially be coalesced with next
        // access!
        currentCoalescingFactor = 1;
      }
      nextAddress = address;
      nextAddress.back() += stride;
    });
    // Account for final run
    coalescingFactor = std::gcd(coalescingFactor, currentCoalescingFactor);
  });

  return coalescingFactor;
}

//===----------------------------------------------------------------------===//
// High-Performance Affine Expression Interpreter
//===----------------------------------------------------------------------===//
//
// This interpreter provides fast evaluation of MLIR AffineExpr by compiling
// expressions into a compact bytecode-style AST that can be evaluated
// efficiently without MLIR infrastructure overhead.
//
// Usage:
//   AffineExprInterpreter interp(affineExpr);
//   int64_t result = interp.evaluate(dimValues);
//
// For evaluating entire maps:
//   AffineMapInterpreter mapInterp(affineMap);
//   SmallVector<int64_t> results = mapInterp.evaluate(dimValues);
//
//===----------------------------------------------------------------------===//

/// Opcodes for the bytecode representation of affine expressions.
enum class AffineExprOpcode : uint8_t {
  Constant, // Push a constant value
  Dim,      // Push a dimension value
  Symbol,   // Push a symbol value
  Add,      // Pop two values, push their sum
  Mul,      // Pop two values, push their product
  Mod,      // Pop two values, push lhs % rhs
  FloorDiv, // Pop two values, push lhs floordiv rhs
  CeilDiv,  // Pop two values, push lhs ceildiv rhs
  // Power-of-2 optimized variants (use bit operations instead of division)
  ModPow2,      // Pop one value, push lhs & (2^operand - 1)
  FloorDivPow2, // Pop one value, push lhs >> operand
  CeilDivPow2,  // Pop one value, push (lhs + (2^operand - 1)) >> operand
  MulPow2,      // Pop one value, push lhs << operand
};

/// Check if a value is a power of 2.
inline bool isPowerOf2(int64_t val) {
  return val > 0 && (val & (val - 1)) == 0;
}

/// Get the log2 of a power of 2 value.
inline int64_t log2Pow2(int64_t val) {
  int64_t log = 0;
  while ((1LL << log) < val) {
    ++log;
  }
  return log;
}

/// A single instruction in the bytecode representation.
/// Uses a compact layout to maximize cache efficiency.
struct AffineExprInstruction {
  AffineExprOpcode opcode;
  // For Constant: the constant value
  // For Dim/Symbol: the index (position)
  // For Pow2 ops: the shift amount (log2 of the power-of-2 constant)
  // For other binary ops: unused (operands come from stack)
  int64_t operand;

  AffineExprInstruction(AffineExprOpcode op, int64_t val = 0)
      : opcode(op), operand(val) {}
};

/// High-performance interpreter for a single AffineExpr.
/// Compiles the expression into a stack-based bytecode representation
/// that can be evaluated efficiently without MLIR overhead.
class AffineExprInterpreter {
public:
  /// Compile an AffineExpr into bytecode for fast repeated evaluation.
  explicit AffineExprInterpreter(mlir::AffineExpr expr) { compile(expr); }

  /// Default constructor for use in containers.
  AffineExprInterpreter() = default;

  /// Evaluate the compiled expression with the given dimension values.
  int64_t
  evaluate(llvm::ArrayRef<int64_t> dimValues,
           [[maybe_unused]] llvm::ArrayRef<int64_t> symbolValues = {}) const {
    llvm::SmallVector<int64_t, 16> stack(instructions_.size());
    return evaluateFast(dimValues, stack);
  }

  /// Fast evaluation using a pre-allocated stack (avoids allocation overhead).
  /// The stack is cleared and reused. Returns the result value.
  int64_t evaluateFast(llvm::ArrayRef<int64_t> dimValues,
                       llvm::MutableArrayRef<int64_t> stack) const {
    size_t sp = 0; // Stack pointer

    for (const auto &inst : instructions_) {
      switch (inst.opcode) {
      case AffineExprOpcode::Constant:
        stack[sp++] = inst.operand;
        break;

      case AffineExprOpcode::Dim:
        stack[sp++] = dimValues[inst.operand];
        break;

      case AffineExprOpcode::Symbol:
        // Symbol support not needed for fast path
        stack[sp++] = 0;
        break;

      case AffineExprOpcode::Add: {
        int64_t rhs = stack[--sp];
        int64_t lhs = stack[--sp];
        stack[sp++] = lhs + rhs;
        break;
      }

      case AffineExprOpcode::Mul: {
        int64_t rhs = stack[--sp];
        int64_t lhs = stack[--sp];
        stack[sp++] = lhs * rhs;
        break;
      }

      case AffineExprOpcode::Mod: {
        int64_t rhs = stack[--sp];
        int64_t lhs = stack[--sp];
        // For positive values (common case), simple mod works
        stack[sp++] = lhs % rhs;
        break;
      }

      case AffineExprOpcode::FloorDiv: {
        int64_t rhs = stack[--sp];
        int64_t lhs = stack[--sp];
        // For positive values (common case), simple division works
        stack[sp++] = lhs / rhs;
        break;
      }

      case AffineExprOpcode::CeilDiv: {
        int64_t rhs = stack[--sp];
        int64_t lhs = stack[--sp];
        stack[sp++] = (lhs + rhs - 1) / rhs;
        break;
      }

      // Power-of-2 optimized operations (use bit operations)
      case AffineExprOpcode::ModPow2: {
        // x % (2^n) = x & ((2^n) - 1)
        int64_t lhs = stack[--sp];
        int64_t mask = (1LL << inst.operand) - 1;
        stack[sp++] = lhs & mask;
        break;
      }

      case AffineExprOpcode::FloorDivPow2: {
        // x / (2^n) = x >> n (for positive x)
        int64_t lhs = stack[--sp];
        stack[sp++] = lhs >> inst.operand;
        break;
      }

      case AffineExprOpcode::CeilDivPow2: {
        // ceil(x / 2^n) = (x + 2^n - 1) >> n
        int64_t lhs = stack[--sp];
        int64_t mask = (1LL << inst.operand) - 1;
        stack[sp++] = (lhs + mask) >> inst.operand;
        break;
      }

      case AffineExprOpcode::MulPow2: {
        // x * (2^n) = x << n
        int64_t lhs = stack[--sp];
        stack[sp++] = lhs << inst.operand;
        break;
      }
      }
    }

    return stack[0];
  }

  /// Check if the interpreter has been compiled with an expression.
  bool isValid() const { return !instructions_.empty(); }

  /// Get the number of instructions (for profiling/debugging).
  size_t getInstructionCount() const { return instructions_.size(); }

  /// Get the maximum stack depth needed for evaluation.
  size_t getMaxStackDepth() const { return instructions_.size(); }

  /// Check if this is a constant expression (no variable dims).
  bool isConstant() const {
    for (const auto &inst : instructions_) {
      if (inst.opcode == AffineExprOpcode::Dim) {
        return false;
      }
    }
    return true;
  }

private:
  llvm::SmallVector<AffineExprInstruction, 8> instructions_;

  /// Recursively compile an AffineExpr into bytecode instructions.
  /// Uses post-order traversal so operands are evaluated before operators.
  void compile(mlir::AffineExpr expr) {
    if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
      instructions_.emplace_back(AffineExprOpcode::Constant,
                                 constExpr.getValue());
      return;
    }

    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
      instructions_.emplace_back(AffineExprOpcode::Dim, dimExpr.getPosition());
      return;
    }

    if (auto symExpr = llvm::dyn_cast<mlir::AffineSymbolExpr>(expr)) {
      instructions_.emplace_back(AffineExprOpcode::Symbol,
                                 symExpr.getPosition());
      return;
    }

    auto binExpr = llvm::cast<mlir::AffineBinaryOpExpr>(expr);

    // Check for power-of-2 optimizations on RHS constant
    if (auto rhsConst =
            llvm::dyn_cast<mlir::AffineConstantExpr>(binExpr.getRHS())) {
      int64_t rhsVal = rhsConst.getValue();
      if (isPowerOf2(rhsVal)) {
        int64_t shift = log2Pow2(rhsVal);
        switch (binExpr.getKind()) {
        case mlir::AffineExprKind::Mod:
          compile(binExpr.getLHS());
          instructions_.emplace_back(AffineExprOpcode::ModPow2, shift);
          return;
        case mlir::AffineExprKind::FloorDiv:
          compile(binExpr.getLHS());
          instructions_.emplace_back(AffineExprOpcode::FloorDivPow2, shift);
          return;
        case mlir::AffineExprKind::CeilDiv:
          compile(binExpr.getLHS());
          instructions_.emplace_back(AffineExprOpcode::CeilDivPow2, shift);
          return;
        case mlir::AffineExprKind::Mul:
          compile(binExpr.getLHS());
          instructions_.emplace_back(AffineExprOpcode::MulPow2, shift);
          return;
        default:
          break; // Fall through to normal compilation
        }
      }
    }

    // Compile operands first (post-order traversal)
    compile(binExpr.getLHS());
    compile(binExpr.getRHS());

    // Then add the operation
    switch (binExpr.getKind()) {
    case mlir::AffineExprKind::Add:
      instructions_.emplace_back(AffineExprOpcode::Add);
      break;
    case mlir::AffineExprKind::Mul:
      instructions_.emplace_back(AffineExprOpcode::Mul);
      break;
    case mlir::AffineExprKind::Mod:
      instructions_.emplace_back(AffineExprOpcode::Mod);
      break;
    case mlir::AffineExprKind::FloorDiv:
      instructions_.emplace_back(AffineExprOpcode::FloorDiv);
      break;
    case mlir::AffineExprKind::CeilDiv:
      instructions_.emplace_back(AffineExprOpcode::CeilDiv);
      break;
    default:
      llvm_unreachable("Unknown affine expression kind");
    }
  }
};

/// High-performance interpreter for an entire AffineMap.
/// Compiles all result expressions for fast batch evaluation.
class AffineMapInterpreter {
public:
  /// Compile an AffineMap into interpreters for all result expressions.
  explicit AffineMapInterpreter(mlir::AffineMap map)
      : numDims_(map.getNumDims()), numSymbols_(map.getNumSymbols()),
        maxStackDepth_(0), originalMap_(map) {
    resultInterpreters_.reserve(map.getNumResults());
    for (auto result : map.getResults()) {
      resultInterpreters_.emplace_back(result);
      maxStackDepth_ = std::max(maxStackDepth_,
                                resultInterpreters_.back().getMaxStackDepth());
    }
  }

  /// Default constructor.
  AffineMapInterpreter() : numDims_(0), numSymbols_(0), maxStackDepth_(0) {}

  /// Evaluate all results of the map with the given dimension values.
  llvm::SmallVector<int64_t, 4>
  evaluate(llvm::ArrayRef<int64_t> dimValues,
           llvm::ArrayRef<int64_t> symbolValues = {}) const {
    llvm::SmallVector<int64_t, 4> results;
    results.reserve(resultInterpreters_.size());
    for (const auto &interp : resultInterpreters_) {
      results.push_back(interp.evaluate(dimValues, symbolValues));
    }
    return results;
  }

  /// Fast evaluation using a pre-allocated stack buffer.
  /// The stack buffer must be at least getMaxStackDepth() elements.
  void evaluateFast(llvm::ArrayRef<int64_t> dimValues,
                    llvm::MutableArrayRef<int64_t> results,
                    llvm::MutableArrayRef<int64_t> stack) const {
    for (size_t i = 0; i < resultInterpreters_.size(); ++i) {
      results[i] = resultInterpreters_[i].evaluateFast(dimValues, stack);
    }
  }

  /// Get the number of results in the map.
  size_t getNumResults() const { return resultInterpreters_.size(); }

  /// Get the number of dimensions expected.
  unsigned getNumDims() const { return numDims_; }

  /// Get the number of symbols expected.
  unsigned getNumSymbols() const { return numSymbols_; }

  /// Get the maximum stack depth needed for any result expression.
  size_t getMaxStackDepth() const { return maxStackDepth_; }

  /// Check if the interpreter has been compiled.
  bool isValid() const { return !resultInterpreters_.empty(); }

  /// Get access to the original map for specialization.
  mlir::AffineMap getMap() const { return originalMap_; }

  /// Store the original map for later specialization.
  void setMap(mlir::AffineMap map) { originalMap_ = map; }

private:
  llvm::SmallVector<AffineExprInterpreter, 4> resultInterpreters_;
  unsigned numDims_;
  unsigned numSymbols_;
  size_t maxStackDepth_;
  mlir::AffineMap originalMap_;
};

/// Threshold for enabling parallel execution (512 elements in shard).
constexpr size_t kParallelShardVolumeThreshold = 512;

/// Get the number of worker threads for parallel execution.
/// Uses hardware concurrency, with a reasonable cap and fallback.
inline unsigned getNumWorkerThreads() {
  unsigned hwThreads = std::thread::hardware_concurrency();
  // Fallback to 4 if hardware_concurrency returns 0 (unable to detect)
  if (hwThreads == 0) {
    hwThreads = 4;
  }
  // Cap at 8 threads to avoid excessive overhead for small workloads
  return std::min(hwThreads, 8u);
}

/// Substitute constant values for grid dimensions and fold the expression.
/// gridDimValues contains the constant values for dimensions 0..gridRank-1.
/// Shard dimensions (gridRank and beyond) are remapped to 0-based indices.
/// Uses recursive AST traversal for simplification.
inline mlir::AffineExpr
substituteGridDimsAndFold(mlir::AffineExpr expr,
                          llvm::ArrayRef<int64_t> gridDimValues,
                          unsigned gridRank, mlir::MLIRContext *context) {
  // Handle constants - already folded
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    return expr;
  }

  // Handle dimension expressions
  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    unsigned pos = dimExpr.getPosition();
    if (pos < gridRank) {
      // Grid dimension - substitute with constant
      return mlir::getAffineConstantExpr(gridDimValues[pos], context);
    }
    // Shard dimension - remap to 0-based index
    return mlir::getAffineDimExpr(pos - gridRank, context);
  }

  // Handle symbols - pass through unchanged
  if (llvm::isa<mlir::AffineSymbolExpr>(expr)) {
    return expr;
  }

  // Handle binary expressions - recurse and fold
  auto binExpr = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
  auto lhs = substituteGridDimsAndFold(binExpr.getLHS(), gridDimValues,
                                       gridRank, context);
  auto rhs = substituteGridDimsAndFold(binExpr.getRHS(), gridDimValues,
                                       gridRank, context);

  // If both are constants, fold the operation
  auto lhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs);
  auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);

  if (lhsConst && rhsConst) {
    int64_t lhsVal = lhsConst.getValue();
    int64_t rhsVal = rhsConst.getValue();
    int64_t result;
    switch (binExpr.getKind()) {
    case mlir::AffineExprKind::Add:
      result = lhsVal + rhsVal;
      break;
    case mlir::AffineExprKind::Mul:
      result = lhsVal * rhsVal;
      break;
    case mlir::AffineExprKind::Mod:
      result = lhsVal % rhsVal;
      break;
    case mlir::AffineExprKind::FloorDiv:
      result = lhsVal / rhsVal;
      break;
    case mlir::AffineExprKind::CeilDiv:
      result = (lhsVal + rhsVal - 1) / rhsVal;
      break;
    default:
      llvm_unreachable("Unknown affine expression kind");
    }
    return mlir::getAffineConstantExpr(result, context);
  }

  // Rebuild the binary expression with simplified operands
  switch (binExpr.getKind()) {
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
    llvm_unreachable("Unknown affine expression kind");
  }
}

/// Substitute constant values for grid dimensions in an affine map.
/// Returns a new map with grid dimensions replaced by constants and folded.
/// The resulting map has shardRank dimensions (renumbered from 0).
inline mlir::AffineMap
substituteGridDimsInMap(mlir::AffineMap map,
                        llvm::ArrayRef<int64_t> gridDimValues,
                        unsigned gridRank, unsigned shardRank) {
  mlir::MLIRContext *context = map.getContext();
  llvm::SmallVector<mlir::AffineExpr> simplifiedResults;
  simplifiedResults.reserve(map.getNumResults());

  for (auto result : map.getResults()) {
    simplifiedResults.push_back(
        substituteGridDimsAndFold(result, gridDimValues, gridRank, context));
  }

  return mlir::AffineMap::get(shardRank, 0, simplifiedResults, context);
}

/// Internal: Process a single grid point and return its coalescing factor.
/// This is extracted to enable parallel execution.
/// Pre-substitutes grid coordinates into affine expressions and folds constants
/// to simplify evaluation.
inline size_t calculateCoalescingFactorForGridPoint(
    mlir::AffineMap map, int64_t gridIdx, llvm::ArrayRef<int64_t> gridShape,
    llvm::ArrayRef<int64_t> gridStrides, llvm::ArrayRef<int64_t> shardShape,
    llvm::ArrayRef<int64_t> outerShardStrides, int64_t outerShardVolume,
    [[maybe_unused]] size_t numDims, [[maybe_unused]] size_t innermostDimIdx,
    int64_t innermostSize, int64_t stride, size_t numResults) {

  size_t shardVolume = volume(shardShape);
  size_t localCoalescingFactor = shardVolume;

  // Compute grid coordinates for this grid point
  llvm::SmallVector<int64_t, 4> gridCoords(gridShape.size());
  for (unsigned j = 0; j < gridShape.size(); ++j) {
    gridCoords[j] = (gridIdx / gridStrides[j]) % gridShape[j];
  }

  // Substitute grid dimensions with constants and fold expressions.
  // This simplifies the map so it only depends on shard dimensions.
  mlir::AffineMap simplifiedMap = substituteGridDimsInMap(
      map, gridCoords, gridShape.size(), shardShape.size());

  // Create interpreter for simplified map (only depends on shard dims now)
  AffineMapInterpreter mapInterp(simplifiedMap);

  // Thread-local buffers - only need shard dimensions now
  llvm::SmallVector<int64_t, 8> shardIndex(shardShape.size(), 0);
  llvm::SmallVector<int64_t, 16> evalStack(mapInterp.getMaxStackDepth());

  // Double-buffer for addresses
  llvm::SmallVector<int64_t, 8> addressBuf0(numResults);
  llvm::SmallVector<int64_t, 8> addressBuf1(numResults);
  int64_t *address = addressBuf0.data();
  int64_t *nextAddress = addressBuf1.data();

  bool nextAddressValid = false;
  size_t currentCoalescingFactor = 0;

  // Helper to compare addresses
  auto addressesEqual = [numResults](const int64_t *a, const int64_t *b) {
    for (size_t i = 0; i < numResults; ++i) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  };

  // Innermost dimension index in shard-only coordinates
  size_t shardInnermostIdx = shardShape.size() - 1;

  // Loop over outer shard dimensions (all but innermost)
  for (int64_t outerShardIdx = 0; outerShardIdx < outerShardVolume;
       ++outerShardIdx) {
    if (localCoalescingFactor == 1) {
      break;
    }

    // Compute outer shard coordinates
    for (unsigned j = 0; j + 1 < shardShape.size(); ++j) {
      shardIndex[j] = (outerShardIdx / outerShardStrides[j]) % shardShape[j];
    }

    // Inner loop over innermost dimension
    for (int64_t inner = 0; inner < innermostSize; ++inner) {
      shardIndex[shardInnermostIdx] = inner;
      mapInterp.evaluateFast(
          shardIndex, llvm::MutableArrayRef<int64_t>(address, numResults),
          evalStack);

      if (!nextAddressValid || addressesEqual(address, nextAddress)) {
        ++currentCoalescingFactor;
      } else {
        localCoalescingFactor =
            std::gcd(localCoalescingFactor, currentCoalescingFactor);
        if (localCoalescingFactor == 1) {
          break;
        }
        currentCoalescingFactor = 1;
      }

      std::swap(address, nextAddress);
      nextAddress[numResults - 1] += stride;
      nextAddressValid = true;
    }

    if (localCoalescingFactor == 1) {
      break;
    }
  }

  // Account for final run
  return std::gcd(localCoalescingFactor, currentCoalescingFactor);
}

/// Calculates the coalescing factor using the high-performance interpreter.
/// This is a drop-in replacement for calculateCoalescingFactor that avoids
/// MLIR infrastructure overhead during evaluation.
/// For large shard volumes (>16K elements), uses parallel execution.
inline size_t calculateCoalescingFactorFast(mlir::AffineMap map,
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
  const size_t numResults = map.getNumResults();

  // Compute grid volume and strides
  int64_t gridVolume = gridShape.empty() ? 1 : volume(gridShape);
  llvm::SmallVector<int64_t, 4> gridStrides;
  if (!gridShape.empty()) {
    gridStrides = calculateStrides(gridShape);
  }

  const size_t numDims = gridShape.size() + shardShape.size();
  const size_t innermostDimIdx = numDims - 1;
  const int64_t innermostSize = shardShape.back();

  // Compute strides for outer shard dimensions
  llvm::SmallVector<int64_t, 8> outerShardStrides;
  int64_t outerShardVolume = 1;
  if (shardShape.size() > 1) {
    auto outerShardShape = shardShape.drop_back();
    outerShardStrides = calculateStrides(outerShardShape);
    outerShardVolume = volume(outerShardShape);
  }

  // Decide whether to use parallel execution
  bool useParallel =
      shardVolume > kParallelShardVolumeThreshold && gridVolume > 1;

  if (!useParallel) {
    // Sequential execution - single grid point or small shard
    size_t coalescingFactor = shardVolume;
    for (int64_t gridIdx = 0; gridIdx < gridVolume; ++gridIdx) {
      if (coalescingFactor == 1) {
        break;
      }
      size_t gridResult = calculateCoalescingFactorForGridPoint(
          map, gridIdx, gridShape, gridStrides, shardShape, outerShardStrides,
          outerShardVolume, numDims, innermostDimIdx, innermostSize, stride,
          numResults);
      coalescingFactor = std::gcd(coalescingFactor, gridResult);
    }
    return coalescingFactor;
  }

  // Parallel execution for large shard volumes
  std::atomic<size_t> globalCoalescingFactor(shardVolume);
  std::atomic<int64_t> nextGridIdx(0);

  auto workerFn = [&]() {
    while (true) {
      // Early exit if coalescing factor is already 1
      if (globalCoalescingFactor.load(std::memory_order_relaxed) == 1) {
        return;
      }

      // Claim next grid index
      int64_t gridIdx = nextGridIdx.fetch_add(1, std::memory_order_relaxed);
      if (gridIdx >= gridVolume) {
        return;
      }

      // Process this grid point
      size_t gridResult = calculateCoalescingFactorForGridPoint(
          map, gridIdx, gridShape, gridStrides, shardShape, outerShardStrides,
          outerShardVolume, numDims, innermostDimIdx, innermostSize, stride,
          numResults);

      // Update global coalescing factor atomically using CAS loop
      size_t current = globalCoalescingFactor.load(std::memory_order_relaxed);
      size_t newVal;
      do {
        newVal = std::gcd(current, gridResult);
        if (newVal == current) {
          break; // No change needed
        }
      } while (!globalCoalescingFactor.compare_exchange_weak(
          current, newVal, std::memory_order_relaxed));
    }
  };

  // Launch worker threads
  llvm::SmallVector<std::thread, 8> workers;
  unsigned numThreads =
      std::min(static_cast<unsigned>(gridVolume), getNumWorkerThreads());
  workers.reserve(numThreads);
  for (unsigned i = 0; i < numThreads; ++i) {
    workers.emplace_back(workerFn);
  }

  // Wait for all workers to complete
  for (auto &worker : workers) {
    worker.join();
  }

  return globalCoalescingFactor.load();
}
} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPUTILS_H
