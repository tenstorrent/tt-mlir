// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::d2m::utils {

// Marker attribute for affine/scf blocking loops in GenericOp regions.
// The attribute value is an i64 block-factor dimension index.
constexpr llvm::StringLiteral kBlockingLoopAttr = "d2m.blocking_loop";

// Return a new RankedTensorType by reblocking its device shape to match a new
// grid shape.
RankedTensorType reblockTensor(RankedTensorType oldTensor,
                               ArrayRef<int64_t> newGridShape);

// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

// Get the largest destination element type used in a region.
// Asserts if no DST-using ops are found.
Type getRegionLargestDstElemType(Region &region);

// Get the largest destination element type used in a region.
// Returns nullptr if no DST-using ops are found.
Type getRegionLargestDstElemTypeOrNull(Region &region);

// Computes dim constraints implied by the indexing maps and shapes. If
// successful, returns a vector of dim constraints for each dimension; a '0'
// indicates that the dimension is not constrained. If the shapes imply
// incompatible constraints, returns std::nullopt.
std::optional<SmallVector<int64_t>>
computeDimConstraints(mlir::ArrayRef<mlir::AffineMap> indexingMaps,
                      mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes);

// Build grid dimension indices from an indexing map. For each result in the
// indexing map, translates arbitrary affine expressions into arith dialect
// operations to compute the index values. This supports all valid affine
// expressions including binary operations (add, mul, floordiv, ceildiv, mod).
SmallVector<Value> buildGridIndices(OpBuilder &builder, Location loc,
                                    AffineMap indexingMap);

// Gets the underlying physical grid shape corresponding to the tensor or
// memref. For views/streams, this 'physical' grid corresponds to the compute
// grid shape used if the tensor/memref was the output of a GenericOp.
SmallVector<int64_t> getPhysicalGridShape(Value tensorOrMemref);

// Finds a 2D grid (y, x) such that y * x = gridVolume. The returned grid aims
// to be as square as possible while respecting the provided target grid shape
// bounds. If either MxN or NxM grids are feasible where M > N, MxN is chosen.
SmallVector<int64_t>
findLegalPhysicalGridForVolume(int64_t gridVolume,
                               ArrayRef<int64_t> targetGridShape);

} // namespace mlir::tt::d2m::utils

#endif
