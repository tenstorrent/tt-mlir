// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m {
class TileBcastOp;
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m::utils {

// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

// Get the largest destination element type used in a region.
Type getRegionLargestDstElemType(Region &region);

// Computes dim constraints implied by the indexing maps and shapes. If
// successful, returns a vector of dim constraints for each dimension; a '0'
// indicates that the dimension is not constrained. If the shapes imply
// incompatible constraints, returns std::nullopt.
std::optional<SmallVector<int64_t>>
computeDimConstraints(mlir::ArrayRef<mlir::AffineMap> indexingMaps,
                      mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes);

/// Trace through the SSA chain to find the affine.load operation.
/// This traces through intermediate compute operations (like tile_bcast) that
/// may be between the load and the consuming compute operation.
///
/// If `outBcastOp` is non-null and a TileBcastOp is encountered during
/// tracing, it will be set to that operation. This allows callers to
/// detect when a load goes through a broadcast without reimplementing
/// the tracing logic.
///
/// @param operand The value to trace through.
/// @param outBcastOp If non-null, will be set to the TileBcastOp if one is
/// encountered during tracing.
/// @return The affine.load operation.
mlir::affine::AffineLoadOp traceToAffineLoad(mlir::Value operand,
                                             TileBcastOp *outBcastOp = nullptr);

} // namespace mlir::tt::d2m::utils

#endif
