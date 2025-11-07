// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

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

  return canonicalToInput.compose(outputToCanonical);
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

// Build semi-affine map from logical indices to physical indices.
// This handles dimension collapse specified by collapsed_intervals.
//
// Example:
//   logical shape: [4, 8, 16]
//   collapsed_intervals: [[0, 2], [2, 3]] - collapse dims 0,1 together
//   physical shape: [32, 16] - product of collapsed dims
//
//   Result: (d0, d1, d2) -> (d0 * 8 + d1, d2).
mlir::AffineMap buildLogicalToPhysicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    mlir::DenseIntElementsAttr collapsedIntervals, mlir::MLIRContext *context) {

  // Normalize intervals to handle negative indices, empty intervals, etc.
  auto normalizedIntervals =
      ttcore::MetalLayoutAttr::normalizeAndFlattenIntervals(
          collapsedIntervals, logicalShape.size());

  assert(normalizedIntervals.size() % 2 == 0 && "Intervals must come in pairs");
  int64_t numIntervals = normalizedIntervals.size() / 2;

  SmallVector<mlir::AffineExpr> physExprs;
  physExprs.reserve(numIntervals);

  for (int64_t i = 0; i < numIntervals; ++i) {
    int64_t start = normalizedIntervals[i * 2];
    int64_t end = normalizedIntervals[i * 2 + 1];

    if (end - start == 1) {
      // Single dimension - just map directly.
      physExprs.push_back(getAffineDimExpr(start, context));
    } else {
      // Multiple dimensions - collapse by computing linear index
      // Result: d_start * stride_{start+1} + d_{start+1} * stride_{start+2} ...
      mlir::AffineExpr collapsed = getAffineConstantExpr(0, context);
      int64_t multiplier = 1;

      for (int64_t d = end - 1; d >= start; --d) {
        mlir::AffineExpr dim = getAffineDimExpr(d, context);
        collapsed = dim * multiplier + collapsed;
        multiplier *= logicalShape[d];
      }

      physExprs.push_back(collapsed);
    }
  }

  return mlir::AffineMap::get(logicalShape.size(), 0, physExprs, context);
}

// Build semi-affine map from physical indices to logical indices (inverse of
// collapse). This expands collapsed physical dimensions back to logical
// dimensions.
//
// Example:
//   logical shape: [32, 64]
//   physical shape: [2048] (collapsed)
//   collapsed intervals: [[0, 2]]
//
//   Result: (d0) -> (d0 floordiv 64, d0 mod 64)
mlir::AffineMap buildPhysicalToLogicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    mlir::DenseIntElementsAttr collapsedIntervals, mlir::MLIRContext *context) {
  auto normalizedIntervals =
      ttcore::MetalLayoutAttr::normalizeAndFlattenIntervals(
          collapsedIntervals, logicalShape.size());

  assert(normalizedIntervals.size() % 2 == 0);
  int64_t numIntervals = normalizedIntervals.size() / 2;

  SmallVector<mlir::AffineExpr> logicalExprs;
  logicalExprs.resize(logicalShape.size());

  for (int64_t i = 0; i < numIntervals; ++i) {
    int64_t start = normalizedIntervals[i * 2];
    int64_t end = normalizedIntervals[i * 2 + 1];

    if (end - start == 1) {
      // Single dimension - direct mapping
      logicalExprs[start] = getAffineDimExpr(i, context);
    } else {
      // Multiple dimensions - expand using mod/floordiv
      mlir::AffineExpr physDim = getAffineDimExpr(i, context);

      int64_t multiplier = 1;
      for (int64_t d = end - 1; d >= start; --d) {
        if (d == end - 1) {
          logicalExprs[d] = physDim % logicalShape[d];
        } else {
          logicalExprs[d] = (physDim.floorDiv(multiplier)) % logicalShape[d];
        }
        multiplier *= logicalShape[d];
      }
    }
  }

  return mlir::AffineMap::get(physicalShape.size(), 0, logicalExprs, context);
}

// Build affine map from device indices to physical indices.
// This reconstructs physical coordinates from grid + shard coordinates.
//
// Example:
//   physical shape: [128, 256]
//   grid shape: [4, 8]
//   shard sizes: [32, 32]
//
//   Result: (d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)
//   where first 2 dims are grid coords, last 2 are shard coords.
mlir::AffineMap buildDeviceToPhysicalMap(ArrayRef<int64_t> physicalShape,
                                         ArrayRef<int64_t> gridShape,
                                         mlir::MLIRContext *context) {
  assert(physicalShape.size() == gridShape.size() &&
         "Physical and grid must have same rank");

  size_t rank = physicalShape.size();
  SmallVector<mlir::AffineExpr> physicalExprs;
  physicalExprs.reserve(rank);

  // Device coords: [grid[0], grid[1], ..., shard[0], shard[1], ...]
  // Physical: physical[i] = grid[i] * shardSize + shard[i]
  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr gridDim = getAffineDimExpr(i, context);
    mlir::AffineExpr shardDim = getAffineDimExpr(rank + i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];

    physicalExprs.push_back(gridDim * shardSize + shardDim);
  }

  return mlir::AffineMap::get(rank * 2, 0, physicalExprs, context);
}

// Build semi-affine map from physical indices to device indices.
// This distributes the physical shape across a grid.
//
// Example:
//   physical shape: [128, 256]
//   grid shape: [4, 8]
//
//   Result: (d0, d1) -> (d0 floordiv 32, d1 floordiv 32, d0 mod 32, d1 mod 32)
//   where shard sizes are [128/4=32, 256/8=32].
mlir::AffineMap buildPhysicalToDeviceMap(ArrayRef<int64_t> physicalShape,
                                         ArrayRef<int64_t> gridShape,
                                         mlir::MLIRContext *context) {
  assert(physicalShape.size() == gridShape.size() &&
         "Physical and grid must have same rank");

  size_t rank = physicalShape.size();
  SmallVector<mlir::AffineExpr> deviceExprs;
  deviceExprs.reserve(rank * 2);

  // First rank results are grid coordinates.
  // Next rank results are shard-local coordinates.
  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = getAffineDimExpr(i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];

    // Grid coordinate: position within the worker grid.
    deviceExprs.push_back(dim.floorDiv(shardSize));
  }

  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = getAffineDimExpr(i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];

    // Shard-local coordinate: position within the shard.
    deviceExprs.push_back(dim % shardSize);
  }

  return mlir::AffineMap::get(rank, 0, deviceExprs, context);
}

// Build pseudo-inverse map from device indices to logical indices.
// This composes the inverses of: device->physical->logical
//
// Note: This is only valid within the logical bounds. Padding regions
// will produce incorrect logical coordinates, but that's acceptable since
// they're never accessed.
mlir::AffineMap
buildDeviceToLogicalMap(mlir::tt::ttcore::MetalLayoutAttr layout,
                        mlir::RankedTensorType tensorType,
                        mlir::MLIRContext *context) {

  ArrayRef<int64_t> logicalShape = layout.getLogicalShape();
  ArrayRef<int64_t> tileShape = ttcore::getTensorTileShapeOrEmpty(tensorType);
  auto physicalShapeVec = layout.getPhysicalShape(tileShape);
  ArrayRef<int64_t> physicalShape = physicalShapeVec;
  ArrayRef<int64_t> gridShape = layout.getGridShape(tensorType);

  assert(physicalShape.size() == gridShape.size() &&
         "Physical and grid must have same rank");

  size_t physRank = physicalShape.size();
  size_t deviceRank = physRank * 2;

  // Step 1: Device back to physical.
  // Reconstruct physical index from (grid_coords, shard_coords).
  // phys[i] = grid[i] * shardSize[i] + shard[i].
  SmallVector<mlir::AffineExpr> physExprs;
  physExprs.reserve(physRank);

  for (size_t i = 0; i < physRank; ++i) {
    mlir::AffineExpr gridDim = getAffineDimExpr(i, context);
    mlir::AffineExpr shardDim = getAffineDimExpr(physRank + i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];

    physExprs.push_back(gridDim * shardSize + shardDim);
  }

  auto deviceToPhysical =
      mlir::AffineMap::get(deviceRank, 0, physExprs, context);

  // Step 2: physical back to  logical (inverse of collapse).
  // This requires "expanding" collapsed dimensions.
  // For tiled tensors, both physical and logical need to be in tile units.
  auto normalizedIntervals = layout.getNormalizedIntervals();

  assert(normalizedIntervals.size() % 2 == 0);
  int64_t numIntervals = normalizedIntervals.size() / 2;

  // Convert logical shape to tile units if tiled
  SmallVector<int64_t> logicalShapeInUnits;
  if (!tileShape.empty()) {
    // Tiled: convert logical shape from scalars to tiles
    logicalShapeInUnits.reserve(logicalShape.size());
    for (size_t i = 0; i < logicalShape.size(); ++i) {
      int64_t tileSize = tileShape[i];
      assert(logicalShape[i] % tileSize == 0 &&
             "Logical shape must be divisible by tile size");
      logicalShapeInUnits.push_back(logicalShape[i] / tileSize);
    }
  } else {
    // Non-tiled: use scalar units
    logicalShapeInUnits.assign(logicalShape.begin(), logicalShape.end());
  }

  SmallVector<mlir::AffineExpr> logicalExprs;
  logicalExprs.resize(logicalShapeInUnits.size());

  for (int64_t i = 0; i < numIntervals; ++i) {
    int64_t start = normalizedIntervals[i * 2];
    int64_t end = normalizedIntervals[i * 2 + 1];

    if (end - start == 1) {
      // Single dimension implies a direct mapping.
      logicalExprs[start] = getAffineDimExpr(i, context);
    } else {
      // Multiple dimensions collapsed implies a need to expand using
      // mod/floordiv.
      mlir::AffineExpr physDim = getAffineDimExpr(i, context);

      // Work backwards through the collapsed dimensions.
      int64_t multiplier = 1;
      for (int64_t d = end - 1; d >= start; --d) {
        if (d == end - 1) {
          // Innermost dimension: just modulo by its size.
          logicalExprs[d] = physDim % logicalShapeInUnits[d];
        } else {
          // Outer dimensions: floordiv by accumulated multiplier, then modulo.
          logicalExprs[d] =
              (physDim.floorDiv(multiplier)) % logicalShapeInUnits[d];
        }
        multiplier *= logicalShapeInUnits[d];
      }
    }
  }

  auto physicalToLogical =
      mlir::AffineMap::get(physRank, 0, logicalExprs, context);

  // Compose: device -> physical -> logical
  auto deviceToLogical = physicalToLogical.compose(deviceToPhysical);

  // If the layout has an existing index map (view), compose through it.
  // The indexAffineMap transforms device coords before the standard
  // device->physical->logical transformation.
  auto existingIndexMap = layout.getIndexAffineMap();
  if (existingIndexMap && !existingIndexMap.isEmpty()) {
    // Compose: modified_device -> device -> physical -> logical.
    deviceToLogical = deviceToLogical.compose(existingIndexMap);
  }

  return deviceToLogical;
}

// Build complete layout transformation from one layout to another.
// Strategy: Use buildDeviceToLogicalMap for both layouts, which already handles
// existing index maps correctly.
mlir::AffineMap
buildLayoutTransformMap(mlir::tt::ttcore::MetalLayoutAttr fromLayout,
                        mlir::RankedTensorType fromType,
                        mlir::tt::ttcore::MetalLayoutAttr toLayout,
                        mlir::RankedTensorType toType) {

  MLIRContext *context = fromLayout.getContext();

  // PRECONDITION: Both layouts must have the same logical shape.
  assert(fromLayout.getLogicalShape() == toLayout.getLogicalShape() &&
         "ToLayoutOp requires same logical shape");

  ArrayRef<int64_t> logicalShape = fromLayout.getLogicalShape();
  ArrayRef<int64_t> fromTileShape = ttcore::getTensorTileShapeOrEmpty(fromType);
  ArrayRef<int64_t> toTileShape = ttcore::getTensorTileShapeOrEmpty(toType);

  // CRITICAL: Both INPUT and OUTPUT must have the same tile shape (or both
  // untiled) for the mapping through logical space to work correctly.
  // Tilize/untilize operations are handled separately in
  // lowerFormatConversionGeneric.
  assert(
      (fromTileShape.empty() == toTileShape.empty()) &&
      "Mapping change requires consistent tiling (both tiled or both untiled)");
  assert((fromTileShape.empty() || fromTileShape == toTileShape) &&
         "Mapping change with tiled tensors requires same tile shape");

  // Use a consistent tile shape for converting logical shape to units.
  // Since both sides must match, we can use either one.
  ArrayRef<int64_t> tileShape = fromTileShape;

  // Get shapes for both layouts
  auto fromPhysicalShapeVec = fromLayout.getPhysicalShape(tileShape);
  ArrayRef<int64_t> fromPhysicalShape = fromPhysicalShapeVec;
  ArrayRef<int64_t> fromGridShape = fromLayout.getGridShape(fromType);

  auto toPhysicalShapeVec = toLayout.getPhysicalShape(tileShape);
  ArrayRef<int64_t> toPhysicalShape = toPhysicalShapeVec;
  ArrayRef<int64_t> toGridShape = toLayout.getGridShape(toType);

  // For tiled tensors, physical shape is in tile units, so we need to convert
  // logical shape to tile units to match. For non-tiled, both are in scalar
  // units.
  SmallVector<int64_t> logicalShapeInUnits;
  if (!tileShape.empty()) {
    logicalShapeInUnits.reserve(logicalShape.size());
    for (size_t i = 0; i < logicalShape.size(); ++i) {
      int64_t tileDim = tileShape[i];
      assert(logicalShape[i] % tileDim == 0 &&
             "Logical shape must be divisible by tile size");
      logicalShapeInUnits.push_back(logicalShape[i] / tileDim);
    }
  } else {
    logicalShapeInUnits.assign(logicalShape.begin(), logicalShape.end());
  }

  // Build OUTPUT device → logical map
  // OUTPUT device → OUTPUT physical
  auto toDeviceToToPhysical =
      buildDeviceToPhysicalMap(toPhysicalShape, toGridShape, context);

  // OUTPUT physical → logical (inverse of collapse)
  auto toPhysicalToLogical =
      buildPhysicalToLogicalMap(logicalShapeInUnits, toPhysicalShape,
                                toLayout.getCollapsedIntervals(), context);

  // Compose: OUTPUT device → OUTPUT physical → logical
  auto toDeviceToLogical = toPhysicalToLogical.compose(toDeviceToToPhysical);

  // Account for existing index map on OUTPUT
  auto toExistingIndexMap = toLayout.getIndexAffineMap();
  if (toExistingIndexMap && !toExistingIndexMap.isEmpty()) {
    toDeviceToLogical = toDeviceToLogical.compose(toExistingIndexMap);
  }

  // Build logical → INPUT device map
  // logical → INPUT physical (collapse)
  auto logicalToFromPhysical =
      buildLogicalToPhysicalMap(logicalShapeInUnits, fromPhysicalShape,
                                fromLayout.getCollapsedIntervals(), context);

  // INPUT physical → INPUT device
  auto fromPhysicalToFromDevice =
      buildPhysicalToDeviceMap(fromPhysicalShape, fromGridShape, context);

  // Compose: logical → INPUT physical → INPUT device
  auto logicalToFromDevice =
      fromPhysicalToFromDevice.compose(logicalToFromPhysical);

  // Simplify before composing with existing index maps to avoid complexity
  // growth
  logicalToFromDevice = mlir::simplifyAffineMap(logicalToFromDevice);

  // If the INPUT has an existing index_map, compose it to handle chained views
  auto fromExistingIndexMap = fromLayout.getIndexAffineMap();
  if (fromExistingIndexMap && !fromExistingIndexMap.isEmpty()) {
    logicalToFromDevice = fromExistingIndexMap.compose(logicalToFromDevice);
    logicalToFromDevice = mlir::simplifyAffineMap(logicalToFromDevice);
  }

  // Compose: OUTPUT device → logical → INPUT device
  auto result = logicalToFromDevice.compose(toDeviceToLogical);

  // Simplify and return
  return mlir::simplifyAffineMap(result);
}

} // namespace mlir::tt::d2m::utils
