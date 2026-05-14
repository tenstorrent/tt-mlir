// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_GRIDSELECTIONUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_GRIDSELECTIONUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m {

/// Complete grid decision for one tensor value.
///
/// selectedGrid is the tensor grid stored in the layout. physicalGrid is the
/// 2D worker-grid extent used to place a virtual grid. layoutGrid is the grid
/// used when computing grid-aware dim alignments for this selected grid.
struct GridDecision {
  llvm::SmallVector<int64_t> selectedGrid;
  llvm::SmallVector<int64_t> targetGrid;
  llvm::SmallVector<int64_t> physicalGrid;
  llvm::SmallVector<int64_t> layoutGrid;

  bool empty() const { return selectedGrid.empty(); }
  bool isVirtual() const { return selectedGrid != physicalGrid; }
};

namespace utils {

GridDecision makeGridDecision(ArrayRef<int64_t> selectedGrid,
                              ArrayRef<int64_t> targetGrid);

// Compute optimal grid shape for a given physical shape and target grid by
// finding the largest grid dimensions that evenly divide the physical shape.
// This ensures maximum utilization of available worker cores while maintaining
// even distribution of work.
llvm::SmallVector<int64_t>
computeOptimalBlockShardedGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetGrid);

// Compute optimal virtual grid shape for a given physical shape and target
// grid by exploring Cartesian products of dimension factors. Returns empty
// vector if utilization is too low and block sharding would do better.
llvm::SmallVector<int64_t>
computeOptimalVirtualGrid(ArrayRef<int64_t> physicalShape,
                          ArrayRef<int64_t> targetGrid);

// Determine whether a tensor should use a virtual grid based on its physical
// shape and the target grid. Rank-2 tensors use virtual grids only when block
// sharding underutilizes the target grid and one dominant tensor axis has
// enough legal grid factors to materially improve over block sharding. ND
// tensors use virtual grids when supported by their layout.
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
// given target grid.
ttcore::MetalLayoutAttr layoutWithOptimalGrid(ttcore::MetalLayoutAttr oldLayout,
                                              ArrayRef<int64_t> targetGrid,
                                              bool ttnnMode);

// Create a new RankedTensorType with the given optimal grid, recomputing the
// device shape and layout accordingly.
mlir::RankedTensorType tensorWithOptimalGrid(mlir::RankedTensorType oldTensor,
                                             ArrayRef<int64_t> targetGrid,
                                             bool ttnnMode,
                                             ArrayRef<int64_t> optimalGrid);

} // namespace utils
} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_UTILS_GRIDSELECTIONUTILS_H
