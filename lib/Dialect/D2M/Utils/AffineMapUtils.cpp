// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/AffineMapUtils.h"

#include "ttmlir/Asserts.h"
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

  if (inputShape == outputShape) {
    return AffineMap::getMultiDimIdentityMap(rank, ctx);
  }

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

AffineMap concatInversePermutationMap(SmallVector<AffineMap> affineMaps,
                                      bool reverse) {
  assert(!affineMaps.empty());

  // Reverse the maps to give output dimensions priority in the inverse
  // permutation.
  if (reverse) {
    affineMaps = llvm::to_vector(llvm::reverse(affineMaps));
  }

  // Concatenate all indexing maps together.
  // Matmul example:
  //   (d0, d1, d2) -> (d0, d2)
  //   (d0, d1, d2) -> (d2, d1)
  //   (d0, d1, d2) -> (d0, d1)
  // Becomes: (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)
  AffineMap concat =
      mlir::concatAffineMaps(affineMaps, affineMaps.front().getContext());

  // Invert the permutation to derive loop bounds from operand shapes.
  // Above example becomes: (d0, d1, d2, d3, d4, d5) -> (d0, d3, d1)
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
      // Single dimension maps directly.
      physExprs.push_back(getAffineDimExpr(start, context));
    } else {
      // Multiple dimensions collapse to a linear index.
      // Form: d_start * stride_{start+1} + d_{start+1} * stride_{start+2} + ...
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
      // Single dimension maps directly.
      logicalExprs[start] = getAffineDimExpr(i, context);
    } else {
      // Multiple collapsed dimensions expand using modulo and floor division.
      mlir::AffineExpr physDim = getAffineDimExpr(i, context);

      int64_t multiplier = 1;
      for (int64_t d = end - 1; d >= start; --d) {
        if (d == end - 1) {
          // Innermost dimension uses modulo.
          logicalExprs[d] = physDim % logicalShape[d];
        } else {
          // Outer dimensions use floor division then modulo.
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

  // Reconstruct physical coordinates from grid and shard coordinates.
  // Device coordinates have form: [grid[0], grid[1], ..., shard[0], shard[1],
  // ...] Physical index: physical[i] = grid[i] * shardSize + shard[i]
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
  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = getAffineDimExpr(i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];
    deviceExprs.push_back(dim.floorDiv(shardSize));
  }

  // Next rank results are shard-local coordinates.
  for (size_t i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = getAffineDimExpr(i, context);
    int64_t shardSize = physicalShape[i] / gridShape[i];
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

  // Step 1: Device coordinates back to physical coordinates.
  // Reconstruct physical index from (grid_coords, shard_coords).
  // Form: phys[i] = grid[i] * shardSize[i] + shard[i]
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

  // Step 2: Physical coordinates back to logical coordinates (inverse of
  // collapse). For tiled tensors, both physical and logical shapes must be in
  // tile units.
  auto normalizedIntervals = layout.getNormalizedIntervals();

  assert(normalizedIntervals.size() % 2 == 0);
  int64_t numIntervals = normalizedIntervals.size() / 2;

  // Convert logical shape to tile units for tiled tensors.
  SmallVector<int64_t> logicalShapeInUnits;
  if (!tileShape.empty()) {
    // Tiled tensors: convert from scalar units to tile units.
    logicalShapeInUnits.reserve(logicalShape.size());
    for (size_t i = 0; i < logicalShape.size(); ++i) {
      int64_t tileSize = tileShape[i];
      assert(logicalShape[i] % tileSize == 0 &&
             "Logical shape must be divisible by tile size");
      logicalShapeInUnits.push_back(logicalShape[i] / tileSize);
    }
  } else {
    // Non-tiled tensors: use scalar units directly.
    logicalShapeInUnits.assign(logicalShape.begin(), logicalShape.end());
  }

  SmallVector<mlir::AffineExpr> logicalExprs;
  logicalExprs.resize(logicalShapeInUnits.size());

  for (int64_t i = 0; i < numIntervals; ++i) {
    int64_t start = normalizedIntervals[i * 2];
    int64_t end = normalizedIntervals[i * 2 + 1];

    if (end - start == 1) {
      // Single dimension maps directly.
      logicalExprs[start] = getAffineDimExpr(i, context);
    } else {
      // Multiple dimensions collapsed; expand using mod/floordiv.
      mlir::AffineExpr physDim = getAffineDimExpr(i, context);

      // Work backwards through the collapsed dimensions.
      int64_t multiplier = 1;
      for (int64_t d = end - 1; d >= start; --d) {
        if (d == end - 1) {
          // Innermost dimension: modulo by its size.
          logicalExprs[d] = physDim % logicalShapeInUnits[d];
        } else {
          // Outer dimensions: floor division by accumulated multiplier, then
          // modulo.
          logicalExprs[d] =
              (physDim.floorDiv(multiplier)) % logicalShapeInUnits[d];
        }
        multiplier *= logicalShapeInUnits[d];
      }
    }
  }

  auto physicalToLogical =
      mlir::AffineMap::get(physRank, 0, logicalExprs, context);

  // Compose: device → physical → logical.
  auto deviceToLogical = physicalToLogical.compose(deviceToPhysical);

  // If the layout has an existing index map (view), compose through it.
  // The index map transforms device coordinates before the standard
  // device→physical→logical transformation.
  auto existingIndexMap = layout.getIndexAffineMap();
  if (existingIndexMap && !existingIndexMap.isEmpty()) {
    // Compose: modified_device → device → physical → logical.
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

  // Precondition: Both layouts must have the same logical shape.
  assert(fromLayout.getLogicalShape() == toLayout.getLogicalShape() &&
         "ToLayoutOp requires same logical shape");

  ArrayRef<int64_t> logicalShape = fromLayout.getLogicalShape();
  ArrayRef<int64_t> fromTileShape = ttcore::getTensorTileShapeOrEmpty(fromType);
  ArrayRef<int64_t> toTileShape = ttcore::getTensorTileShapeOrEmpty(toType);

  // Both input and output must have the same tile shape (or both untiled) for
  // the mapping through logical space to work correctly. Tilize/untilize
  // operations are handled separately in lowerFormatConversionGeneric.
  assert(
      (fromTileShape.empty() == toTileShape.empty()) &&
      "Mapping change requires consistent tiling (both tiled or both untiled)");
  assert((fromTileShape.empty() || fromTileShape == toTileShape) &&
         "Mapping change with tiled tensors requires same tile shape");

  // For collapse/expand maps, always work in SCALAR units.
  // The collapse describes the relationship between logical shape (scalars,
  // unpadded) and physical shape (scalars, padded). This works for both tiled
  // and scalar tensors.
  auto fromPhysicalShapeVec = fromLayout.getPhysicalShape({}); // Scalars
  ArrayRef<int64_t> fromPhysicalShape = fromPhysicalShapeVec;

  auto toPhysicalShapeVec = toLayout.getPhysicalShape({}); // Scalars
  ArrayRef<int64_t> toPhysicalShape = toPhysicalShapeVec;

  // Logical shape is always in scalars
  ArrayRef<int64_t> logicalShapeInUnits = logicalShape;

  // Grid shape is the number of shards, independent of units.
  // The shard size will be computed as physical_shape / grid_shape in the
  // correct units (scalars).
  ArrayRef<int64_t> fromGridShape = fromLayout.getGridShape(fromType);
  ArrayRef<int64_t> toGridShape = toLayout.getGridShape(toType);

  // DEBUG: Print shapes for tracing
  llvm::errs() << "DEBUG buildLayoutTransformMap:\n";
  llvm::errs() << "  Logical shape: [";
  for (size_t i = 0; i < logicalShape.size(); i++) {
    llvm::errs() << logicalShape[i];
    if (i + 1 < logicalShape.size()) {
      llvm::errs() << ", ";
    }
  }
  llvm::errs() << "]\n";

  llvm::errs() << "  FROM (input) physical: [";
  for (size_t i = 0; i < fromPhysicalShape.size(); i++) {
    llvm::errs() << fromPhysicalShape[i];
    if (i + 1 < fromPhysicalShape.size()) {
      llvm::errs() << ", ";
    }
  }
  llvm::errs() << "], grid: [";
  for (size_t i = 0; i < fromGridShape.size(); i++) {
    llvm::errs() << fromGridShape[i];
    if (i + 1 < fromGridShape.size()) {
      llvm::errs() << ", ";
    }
  }
  llvm::errs() << "]\n";

  llvm::errs() << "  TO (output) physical: [";
  for (size_t i = 0; i < toPhysicalShape.size(); i++) {
    llvm::errs() << toPhysicalShape[i];
    if (i + 1 < toPhysicalShape.size()) {
      llvm::errs() << ", ";
    }
  }
  llvm::errs() << "], grid: [";
  for (size_t i = 0; i < toGridShape.size(); i++) {
    llvm::errs() << toGridShape[i];
    if (i + 1 < toGridShape.size()) {
      llvm::errs() << ", ";
    }
  }
  llvm::errs() << "]\n";

  // Build OUTPUT device → logical map.
  // OUTPUT device → OUTPUT physical.
  auto toDeviceToToPhysical =
      buildDeviceToPhysicalMap(toPhysicalShape, toGridShape, context);
  llvm::errs() << "  TO device→physical: " << toDeviceToToPhysical << "\n";

  // OUTPUT physical → logical (inverse of collapse).
  auto toPhysicalToLogical =
      buildPhysicalToLogicalMap(logicalShapeInUnits, toPhysicalShape,
                                toLayout.getCollapsedIntervals(), context);
  llvm::errs() << "  TO physical→logical: " << toPhysicalToLogical << "\n";

  // Compose: OUTPUT device → OUTPUT physical → logical.
  auto toDeviceToLogical = toPhysicalToLogical.compose(toDeviceToToPhysical);
  llvm::errs() << "  TO device→logical (composed): " << toDeviceToLogical
               << "\n";

  // Account for existing index map on OUTPUT.
  auto toExistingIndexMap = toLayout.getIndexAffineMap();
  if (toExistingIndexMap && !toExistingIndexMap.isEmpty()) {
    toDeviceToLogical = toDeviceToLogical.compose(toExistingIndexMap);
  }

  // Build logical → INPUT device map.
  // logical → INPUT physical (collapse).
  auto logicalToFromPhysical =
      buildLogicalToPhysicalMap(logicalShapeInUnits, fromPhysicalShape,
                                fromLayout.getCollapsedIntervals(), context);
  llvm::errs() << "  FROM logical→physical: " << logicalToFromPhysical << "\n";

  // INPUT physical → INPUT device.
  auto fromPhysicalToFromDevice =
      buildPhysicalToDeviceMap(fromPhysicalShape, fromGridShape, context);
  llvm::errs() << "  FROM physical→device: " << fromPhysicalToFromDevice
               << "\n";

  // Compose: logical → INPUT physical → INPUT device.
  auto logicalToFromDevice =
      fromPhysicalToFromDevice.compose(logicalToFromPhysical);
  llvm::errs() << "  FROM logical→device (composed): " << logicalToFromDevice
               << "\n";

  // Simplify before composing with existing index maps to avoid exponential
  // complexity growth.
  logicalToFromDevice = mlir::simplifyAffineMap(logicalToFromDevice);
  llvm::errs() << "  FROM logical→device (simplified): " << logicalToFromDevice
               << "\n";

  // If the INPUT has an existing index_map, compose it to handle chained views.
  auto fromExistingIndexMap = fromLayout.getIndexAffineMap();
  if (fromExistingIndexMap && !fromExistingIndexMap.isEmpty()) {
    logicalToFromDevice = fromExistingIndexMap.compose(logicalToFromDevice);
    logicalToFromDevice = mlir::simplifyAffineMap(logicalToFromDevice);
  }

  // Compose: OUTPUT device → logical → INPUT device.
  auto result = logicalToFromDevice.compose(toDeviceToLogical);
  llvm::errs() << "  FINAL (before simplify): " << result << "\n";

  // Simplify and return.
  result = mlir::simplifyAffineMap(result);
  llvm::errs() << "  FINAL (simplified): " << result << "\n";
  return result;
}

} // namespace mlir::tt::d2m::utils
