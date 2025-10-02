// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::tt::d2m::utils {

// Calculate a reblock affine map between tensor shapes.
mlir::AffineMap calculateReblockMap(ArrayRef<int64_t> fromTensorShape,
                                    ArrayRef<int64_t> toTensorShape,
                                    mlir::MLIRContext *context);

// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

Type getRegionLargestDstElemType(Region &region);

} // namespace mlir::tt::d2m::utils

#endif
