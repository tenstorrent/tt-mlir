// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_GRIDSELECTIONUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_GRIDSELECTIONUTILS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

namespace utils {

// Walk back through any chain of ViewLayoutOp producers and return the
// ToLayoutOp that feeds them. Returns a null ToLayoutOp if `operand` is not
// produced through at least one view, or if the value behind the views is not
// a ToLayoutOp.
d2m::ToLayoutOp getToLayoutProducerBehindViews(mlir::Value operand);

// Walk forward through layout-bridge ops until reaching a tiled ToLayoutOp
// consumer. Returns the tile shape, or empty if no tiled consumer is reachable.
llvm::SmallVector<int64_t>
findDownstreamTiledToLayoutTileShape(mlir::Value value);

// Walk backward through layout-bridge producers until reaching a tiled tensor.
// Returns the tile shape, or empty if the producer chain remains scalar.
llvm::SmallVector<int64_t>
findUpstreamTiledLayoutBridgeTileShape(mlir::Value value);

// Compute the size of a single collapsed-interval row across the given
// logical-shape range, propagating alignment from the innermost dim outward.
int64_t computeCollapsedIntervalSize(ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> alignments,
                                     int64_t intervalStart,
                                     int64_t intervalEnd);

// Compute optimal grid shape for a given physical shape and target grid by
// finding the largest grid dimensions that evenly divide the physical shape.
// This ensures maximum utilization of available worker cores while maintaining
// even distribution of work.
llvm::SmallVector<int64_t>
computeOptimalBlockShardedGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetGrid);

// Compute optimal virtual grid shape for a given physical shape and target
// grid. For ND tensors, explores Cartesian product of dimension factors.
// For 2D tensors, finds the largest factor of the sharded dimension. Returns
// empty vector if utilization is too low (signals fallback to block sharding).
llvm::SmallVector<int64_t>
computeOptimalVirtualGrid(ArrayRef<int64_t> physicalShape,
                          ArrayRef<int64_t> targetGrid);

// Determine whether a tensor should use a virtual grid based on its physical
// shape and the target grid. Returns true when block sharding yields low grid
// utilization or when the tensor is ND.
bool shouldImplementAsVirtualGrid(mlir::RankedTensorType tensorType,
                                  ArrayRef<int64_t> physicalShape,
                                  ArrayRef<int64_t> targetGrid);

// Compute the optimal grid for a tensor, choosing between virtual grid and
// block sharding based on heuristics.
llvm::SmallVector<int64_t> computeOptimalGrid(mlir::RankedTensorType tensorType,
                                              ArrayRef<int64_t> physicalShape,
                                              ArrayRef<int64_t> targetGrid);

// Compute physical shape for a MetalLayoutAttr. In TTNN mode, returns the raw
// physical shape without alignment adjustments. Otherwise, computes grid-aware
// dimension alignments and derives the physical shape (always tile-aligned).
llvm::SmallVector<int64_t> computePhysicalShape(mlir::Value operand,
                                                ArrayRef<int64_t> targetGrid,
                                                bool ttnnMode);

// Create a new MetalLayoutAttr with grid-aware dimension alignments for the
// given selected grid. The tile shape is empty for row-major tensors.
ttcore::MetalLayoutAttr layoutWithOptimalGrid(ttcore::MetalLayoutAttr oldLayout,
                                              ArrayRef<int64_t> selectedGrid,
                                              bool ttnnMode,
                                              ArrayRef<int64_t> tileShape);

// Create a new RankedTensorType with the given optimal grid, recomputing the
// device shape and layout accordingly.
mlir::RankedTensorType
tensorWithOptimalGrid(mlir::RankedTensorType oldTensor, bool ttnnMode,
                      ArrayRef<int64_t> optimalGrid,
                      ArrayRef<int64_t> paddingTileShape = {});

} // namespace utils
} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_GRIDSELECTIONUTILS_H
