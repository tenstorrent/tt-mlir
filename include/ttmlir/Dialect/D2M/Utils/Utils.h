// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m::utils {

// Calculate a reblock affine map between tensor shapes.
mlir::AffineMap calculateReblockMap(ArrayRef<int64_t> fromTensorShape,
                                    ArrayRef<int64_t> toTensorShape,
                                    mlir::MLIRContext *context);

// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

Type getRegionLargestDstElemType(Region &region);

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
// Note: Does NOT handle padding - padding affects shapes, not index mapping.
mlir::AffineMap buildLogicalToPhysicalMap(
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> physicalShape,
    mlir::DenseIntElementsAttr collapsedIntervals, mlir::MLIRContext *context);

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

// Build the full transformation map from one layout to another.
// Both layouts must have the same logical shape (ToLayoutOp invariant).
// Result maps: fromLayout device coords -> toLayout device coords
mlir::AffineMap
buildLayoutTransformMap(mlir::tt::ttcore::MetalLayoutAttr fromLayout,
                        mlir::RankedTensorType fromType,
                        mlir::tt::ttcore::MetalLayoutAttr toLayout,
                        mlir::RankedTensorType toType);

} // namespace mlir::tt::d2m::utils

#endif
