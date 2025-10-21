// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"

namespace mlir::tt::d2m::utils {

// Calculate a reblock affine map between tensor shapes.
mlir::AffineMap calculateReblockMap(ArrayRef<int64_t> fromTensorShape,
                                    ArrayRef<int64_t> toTensorShape,
                                    mlir::MLIRContext *context);

// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

// Trace IR to find underlying physical (non-view) tensor/memref
Value getPhysicalTensorOrMemref(mlir::Value tensorOrMemref);

// Trace IR to find underlying physical (non-view) tensor/memref and return its
// grid shape
llvm::SmallVector<int64_t> getPhysicalGridShape(mlir::Value tensorOrMemref);

// Get grid shape of a tensor or memref using layout attribute
llvm::SmallVector<int64_t> getGridShape(mlir::Value tensorOrMemref);

// Get device layout interface if it exists
ttcore::DeviceLayoutInterface getDeviceLayoutInterfaceIfExists(mlir::Value tensorOrMemref);

} // namespace mlir::tt::d2m::utils

#endif
