// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::d2m::utils {

<<<<<<< HEAD
// Calculate a reblock affine map between tensor shapes.
mlir::AffineMap calculateReblockMap(mlir::ArrayRef<int64_t> fromTensorShape,
                                    mlir::ArrayRef<int64_t> toTensorShape,
                                    mlir::MLIRContext *context);

// Calculate a reblock affine map given a shape and new grid shape.
// Returns the new tensor shape and the reblock affine map.
std::pair<mlir::SmallVector<int64_t>, mlir::AffineMap>
calculateReblockMapForGrid(mlir::ArrayRef<int64_t> tensorShape,
                           mlir::ArrayRef<int64_t> newGridShape,
                           mlir::MLIRContext *context);

=======
>>>>>>> 0eddd4ebc (big cleanup, commit to always materializing view via GenericOp w/ DMA, tests are passing locally)
// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

// Get the largest destination element type used in a region.
Type getRegionLargestDstElemType(Region &region);

} // namespace mlir::tt::d2m::utils

#endif
