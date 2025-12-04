// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m::utils {

// Return a new RankedTensorType by reblocking its device shape to match a new
// grid shape.
RankedTensorType reblockTensor(RankedTensorType oldTensor,
                               ArrayRef<int64_t> newGridShape);

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

} // namespace mlir::tt::d2m::utils

#endif
