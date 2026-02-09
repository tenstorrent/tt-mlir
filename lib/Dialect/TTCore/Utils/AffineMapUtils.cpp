// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/Utils/AffineMapUtils.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttcore::utils {

// Compute aligned strides for a collapsed interval.
// Returns strides from innermost to outermost.
// E.g., for dims [7, 41, 43] with alignments [256, 1, 64]:
//   aligned innermost = 64
//   cumulative [0,1] = alignUp(41 * 64, 1) = 2624
//   cumulative [0,2] = alignUp(7 * 2624, 256) = 18432
//   strides = [2624, 64, 1] (for dims 0, 1, 2)
static SmallVector<int64_t>
computeAlignedStrides(ArrayRef<int64_t> logicalShape,
                      ArrayRef<int64_t> alignments, int64_t start,
                      int64_t end) {
  SmallVector<int64_t> strides(end - start);

  // Compute cumulative aligned products from innermost outward.
  int64_t cumulative = 1;
  for (int64_t d = end - 1; d >= start; --d) {
    strides[d - start] = cumulative;
    int64_t aligned =
        ttmlir::utils::alignUp(logicalShape[d] * cumulative, alignments[d]);
    cumulative = aligned;
  }

  return strides;
}

// Build semi-affine map from logical indices to physical indices.
// This handles dimension collapse specified by collapsed_intervals.
// Uses aligned strides to correctly handle padding regions.
//
// Example:
//   logical shape: [4, 8, 16]
//   collapsed_intervals: [[0, 2], [2, 3]] - collapse dims 0,1 together
//   physical shape: [32, 16] - product of collapsed dims
//
//   Result: (d0, d1, d2) -> (d0 * 8 + d1, d2).
mlir::AffineMap buildLogicalToPhysicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    ArrayRef<int64_t> alignments, mlir::DenseIntElementsAttr collapsedIntervals,
    mlir::MLIRContext *context) {

  // Normalize intervals to handle negative indices, empty intervals, etc.
  auto normalizedIntervals = MetalLayoutAttr::normalizeAndFlattenIntervals(
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
      // Multiple dimensions collapse to a linear index using aligned strides.
      // Form: d_start * stride_{start} + d_{start+1} * stride_{start+1} + ...
      auto strides =
          computeAlignedStrides(logicalShape, alignments, start, end);

      mlir::AffineExpr collapsed = getAffineConstantExpr(0, context);
      for (int64_t d = end - 1; d >= start; --d) {
        mlir::AffineExpr dim = getAffineDimExpr(d, context);
        collapsed = dim * strides[d - start] + collapsed;
      }

      physExprs.push_back(collapsed);
    }
  }

  return mlir::AffineMap::get(logicalShape.size(), 0, physExprs, context);
}

// Build semi-affine map from physical indices to logical indices (inverse of
// collapse). This expands collapsed physical dimensions back to logical
// dimensions. Uses aligned strides to correctly handle padding regions.
//
// Example:
//   logical shape: [32, 64]
//   physical shape: [2048] (collapsed)
//   collapsed intervals: [[0, 2]]
//
//   Result: (d0) -> (d0 floordiv 64, d0 mod 64)
mlir::AffineMap buildPhysicalToLogicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    ArrayRef<int64_t> alignments, mlir::DenseIntElementsAttr collapsedIntervals,
    mlir::MLIRContext *context) {
  auto normalizedIntervals = MetalLayoutAttr::normalizeAndFlattenIntervals(
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
      // Use aligned strides to handle padding correctly.
      mlir::AffineExpr physDim = getAffineDimExpr(i, context);
      auto strides =
          computeAlignedStrides(logicalShape, alignments, start, end);

      for (int64_t d = end - 1; d >= start; --d) {
        int64_t stride = strides[d - start];
        if (d == end - 1) {
          // Innermost dimension: use aligned stride for modulo.
          int64_t alignedDim =
              ttmlir::utils::alignUp(logicalShape[d], alignments[d]);
          logicalExprs[d] = physDim % alignedDim;
        } else if (d == start) {
          // Outermost dimension: just divide by stride, no modulo needed.
          logicalExprs[d] = physDim.floorDiv(stride);
        } else {
          // Middle dimensions: divide by stride, then modulo by range.
          // Range = stride[d-1] / stride[d] = how many values this dim spans.
          int64_t outerStride = strides[d - start - 1];
          int64_t range = outerStride / stride;
          logicalExprs[d] = (physDim.floorDiv(stride)) % range;
        }
      }
    }
  }

  return mlir::AffineMap::get(physicalShape.size(), 0, logicalExprs, context);
}

// Build pseudo-inverse map from device indices to logical indices.
// This composes the inverses of: device->physical->logical
//
// Note: This is only valid within the logical bounds. Padding regions
// will produce incorrect logical coordinates, but that's acceptable since
// they're never accessed.
mlir::AffineMap buildDeviceToLogicalMap(MetalLayoutAttr layout,
                                        mlir::RankedTensorType tensorType,
                                        mlir::MLIRContext *context) {

  ArrayRef<int64_t> logicalShape = layout.getLogicalShape();
  ArrayRef<int64_t> tileShape = getTensorTileShapeOrEmpty(tensorType);
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
      // Use aligned strides to correctly handle padding.
      mlir::AffineExpr physDim = getAffineDimExpr(i, context);
      auto alignments = layout.getDimAlignments();
      auto strides =
          computeAlignedStrides(logicalShapeInUnits, alignments, start, end);

      for (int64_t d = end - 1; d >= start; --d) {
        int64_t stride = strides[d - start];
        if (d == end - 1) {
          // Innermost dimension: use aligned stride for modulo.
          int64_t alignedDim =
              ttmlir::utils::alignUp(logicalShapeInUnits[d], alignments[d]);
          logicalExprs[d] = physDim % alignedDim;
        } else if (d == start) {
          // Outermost dimension: just divide by stride, no modulo needed.
          logicalExprs[d] = physDim.floorDiv(stride);
        } else {
          // Middle dimensions: divide by stride, then modulo by range.
          int64_t outerStride = strides[d - start - 1];
          int64_t range = outerStride / stride;
          logicalExprs[d] = (physDim.floorDiv(stride)) % range;
        }
      }
    }
  }

  auto physicalToLogical =
      mlir::AffineMap::get(physRank, 0, logicalExprs, context);

  // Compose: device → physical → logical.
  auto deviceToLogical = physicalToLogical.compose(deviceToPhysical);

  return deviceToLogical;
}

// Build complete layout transformation from one layout to another.
// Strategy: Use buildDeviceToLogicalMap for both layouts, which already handles
// existing index maps correctly.
mlir::AffineMap buildLayoutTransformMap(MetalLayoutAttr fromLayout,
                                        mlir::RankedTensorType fromType,
                                        MetalLayoutAttr toLayout,
                                        mlir::RankedTensorType toType) {
  MLIRContext *context = fromLayout.getContext();

  // Precondition: Both layouts must have the same logical shape.
  assert(fromLayout.getLogicalShape() == toLayout.getLogicalShape() &&
         "ToLayoutOp requires same logical shape");

  ArrayRef<int64_t> logicalShape = fromLayout.getLogicalShape();
  ArrayRef<int64_t> fromTileShape = getTensorTileShapeOrEmpty(fromType);
  ArrayRef<int64_t> toTileShape = getTensorTileShapeOrEmpty(toType);

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

  // Logical shape is always in scalars.
  ArrayRef<int64_t> logicalShapeInUnits = logicalShape;

  // Grid shape is the number of shards, independent of units.
  // The shard size will be computed as physical_shape / grid_shape in the
  // correct units (scalars).
  ArrayRef<int64_t> fromGridShape = fromLayout.getGridShape(fromType);
  ArrayRef<int64_t> toGridShape = toLayout.getGridShape(toType);

  // Build OUTPUT device → logical map.
  // OUTPUT device → OUTPUT physical.
  auto toDeviceToToPhysical = ttmlir::utils::buildDeviceToPhysicalMap(
      toPhysicalShape, toGridShape, context);

  // OUTPUT physical → logical (inverse of collapse).
  auto toAlignments = toLayout.getDimAlignments();
  auto toPhysicalToLogical = buildPhysicalToLogicalMap(
      logicalShapeInUnits, toPhysicalShape, toAlignments,
      toLayout.getCollapsedIntervals(), context);

  // Compose: OUTPUT device → OUTPUT physical → logical.
  auto toDeviceToLogical = toPhysicalToLogical.compose(toDeviceToToPhysical);

  // Build logical → INPUT device map.
  // logical → INPUT physical (collapse).
  auto fromAlignments = fromLayout.getDimAlignments();
  auto logicalToFromPhysical = buildLogicalToPhysicalMap(
      logicalShapeInUnits, fromPhysicalShape, fromAlignments,
      fromLayout.getCollapsedIntervals(), context);

  // INPUT physical → INPUT device.
  auto fromPhysicalToFromDevice = ttmlir::utils::buildPhysicalToDeviceMap(
      fromPhysicalShape, fromGridShape, context);

  // Compose: logical → INPUT physical → INPUT device.
  auto logicalToFromDevice =
      fromPhysicalToFromDevice.compose(logicalToFromPhysical);

  // Simplify before composing with existing index maps to avoid exponential
  // complexity growth.
  logicalToFromDevice = mlir::simplifyAffineMap(logicalToFromDevice);

  // NOTE: Do NOT compose the INPUT index_map here. The index_map (core
  // virtualization map) will be composed later in DeviceAttr::getMemoryMap
  // when generating the actual DMA. Composing it here would apply it twice,
  // causing incorrect virtual-to-physical coordinate translation.

  // Compose: OUTPUT device → logical → INPUT device.
  auto result = logicalToFromDevice.compose(toDeviceToLogical);

  // Simplify and return.
  return mlir::simplifyAffineMap(result);
}

} // namespace mlir::tt::ttcore::utils
