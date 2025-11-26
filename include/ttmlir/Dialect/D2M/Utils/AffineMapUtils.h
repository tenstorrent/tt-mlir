// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_AFFINEMAPUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_AFFINEMAPUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m::utils {

// Calculate a reblock affine map between tensor shapes.
mlir::AffineMap calculateReblockMap(ArrayRef<int64_t> fromTensorShape,
                                    ArrayRef<int64_t> toTensorShape,
                                    mlir::MLIRContext *context);

// Calculate a reblock affine map given a shape and new grid shape.
// Returns the new tensor shape and the reblock affine map.
std::pair<mlir::SmallVector<int64_t>, mlir::AffineMap>
calculateReblockMapForGrid(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::ArrayRef<int64_t> newGridShape,
                           mlir::MLIRContext *context);

// This routine concatenates the provided affine maps together and then inverts
// the map which is a convenient routine for deriving concrete iterator values.
//
// Using matmul maps for example:
//   (d0, d1, d2) -> (d0, d2)
//   (d0, d1, d2) -> (d2, d1)
//   (d0, d1, d2) -> (d0, d1)
//
//   1. If reverse is set, it will reverse the provided affine maps first.  This
//      is useful for establishing a priority, in most cases thus far it is
//      required that the output operand to a generic gets priority for
//      calculating block factors:
//        (d0, d1, d2) -> (d0, d1)
//        (d0, d1, d2) -> (d2, d1)
//        (d0, d1, d2) -> (d0, d2)
//   2. Concat all of the indexing maps together:
//        (d0, d1, d2) -> (d0, d1, d2, d1, d0, d2)
//   3. Invert the permutation, remapping the results to input iterators:
//        (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)
AffineMap concatInversePermutationMap(SmallVector<AffineMap> affineMaps,
                                      bool reverse);

// Layout transformation utilities:
// These functions build semi-affine maps for the different stages of layout
// transformations. The overall transformation chain is:
//   logical -> physical -> device -> view
// where:
//   - logical: The original tensor shape (shared across all layouts)
//   - physical: After collapse and padding (shape may change per layout)
//   - device: After distribution across grid (includes grid dimensions)
//   - view: After applying index map (optional reinterpretation)

// Build semi-affine map: logical indices -> physical indices
// Handles collapse of dimensions (based on collapsed_intervals).
// Uses aligned strides to correctly handle padding regions.
mlir::AffineMap buildLogicalToPhysicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    ArrayRef<int64_t> alignments, mlir::DenseIntElementsAttr collapsedIntervals,
    mlir::MLIRContext *context);

// Build semi-affine map: physical indices -> logical indices (inverse of
// collapse). Expands collapsed physical dimensions back to logical dimensions.
// Uses aligned strides to correctly handle padding regions.
mlir::AffineMap buildPhysicalToLogicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    ArrayRef<int64_t> alignments, mlir::DenseIntElementsAttr collapsedIntervals,
    mlir::MLIRContext *context);

// Build affine map: device indices -> physical indices
// Reconstructs physical coordinates from grid + shard coordinates.
// Result has form: (grid_dims..., shard_dims...) -> (phys_dims...)
mlir::AffineMap buildDeviceToPhysicalMap(ArrayRef<int64_t> physicalShape,
                                         ArrayRef<int64_t> gridShape,
                                         mlir::MLIRContext *context);

// Build semi-affine map: physical indices -> device indices
// Maps from collapsed/padded shape to grid-distributed shape.
// Result has form: (phys_dims...) -> (grid_dims..., shard_dims...)
mlir::AffineMap buildPhysicalToDeviceMap(ArrayRef<int64_t> physicalShape,
                                         ArrayRef<int64_t> gridShape,
                                         mlir::MLIRContext *context);

// Build semi-affine map: device indices -> logical indices (pseudo-inverse)
// This is the "inverse" of the logical->physical->device chain.
// Only valid within the logical region (padding regions produce undefined
// results, which is acceptable since they're never accessed).
mlir::AffineMap
buildDeviceToLogicalMap(mlir::tt::ttcore::MetalLayoutAttr layout,
                        mlir::RankedTensorType tensorType,
                        mlir::MLIRContext *context);

// Build the full transformation map for a view operation.
// Both layouts must have the same logical shape (ToLayoutOp invariant).
// Result maps: toLayout device coords -> fromLayout device coords
// (i.e., for each output coord, where to fetch input data from)
mlir::AffineMap
buildLayoutTransformMap(mlir::tt::ttcore::MetalLayoutAttr fromLayout,
                        mlir::RankedTensorType fromType,
                        mlir::tt::ttcore::MetalLayoutAttr toLayout,
                        mlir::RankedTensorType toType);

} // namespace mlir::tt::d2m::utils

#endif
