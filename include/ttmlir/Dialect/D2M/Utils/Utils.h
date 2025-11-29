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
AffineMap concatInversePermutationMap(mlir::ArrayRef<AffineMap> affineMaps,
                                      bool reverse);

// Traces IR to find underlying physical (non-view) tensor/memref.
Value getPhysicalTensorOrMemref(mlir::Value tensorOrMemref);

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
