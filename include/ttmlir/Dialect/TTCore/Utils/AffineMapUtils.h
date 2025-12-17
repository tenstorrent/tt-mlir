// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_UTILS_AFFINEMAPUTILS_H
#define TTMLIR_DIALECT_TTCORE_UTILS_AFFINEMAPUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttcore::utils {

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

// Build semi-affine map: device indices -> logical indices (pseudo-inverse)
// This is the "inverse" of the logical->physical->device chain.
// Only valid within the logical region (padding regions produce undefined
// results, which is acceptable since they're never accessed).
mlir::AffineMap buildDeviceToLogicalMap(MetalLayoutAttr layout,
                                        mlir::RankedTensorType tensorType,
                                        mlir::MLIRContext *context);

// Build the full transformation map for a view operation.
// Both layouts must have the same logical shape (ToLayoutOp invariant).
// Result maps: toLayout device coords -> fromLayout device coords
// (i.e., for each output coord, where to fetch input data from)
mlir::AffineMap buildLayoutTransformMap(MetalLayoutAttr fromLayout,
                                        mlir::RankedTensorType fromType,
                                        MetalLayoutAttr toLayout,
                                        mlir::RankedTensorType toType);

} // namespace mlir::tt::ttcore::utils

#endif // TTMLIR_DIALECT_TTCORE_UTILS_AFFINEMAPUTILS_H
