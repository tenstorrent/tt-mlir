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

// Get grid shape of a tensor or memref using its DeviceLayoutInterface.
llvm::SmallVector<int64_t> getGridShape(mlir::Value tensorOrMemref);

// Returns true if the tensor/memref has a virtual grid (i.e. non trivial core
// virtualization map defined in underlying ShardLayoutAttr).
bool hasVirtualGrid(mlir::Value tensorOrMemref);

// Traces IR to find underlying physical (non-view) tensor/memref.
Value getPhysicalTensorOrMemref(mlir::Value tensorOrMemref);

// Trace IR to find underlying physical (non-view) tensor/memref and return its
// grid shape.
llvm::SmallVector<int64_t> getPhysicalGridShape(mlir::Value tensorOrMemref);

// Get core virtualization map associated with a tensor/memref's underlying
// physical implementation. The input through IR to find its underlying physical
// tensor/memref.
AffineMap getCoreVirtualizationMap(mlir::Value tensorOrMemref);

} // namespace mlir::tt::d2m::utils

#endif
