// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace mlir::tt::ttnn::optimizer_utils {

/// Returns true if the op produces a ranked tensor result that the layout
/// optimizer can assign a memory layout to.
inline bool opHasTensorResult(mlir::Operation *op) {
  if (op->getNumResults() == 0) {
    return false;
  }
  if (!llvm::isa<mlir::RankedTensorType>(op->getResult(0).getType())) {
    return false;
  }
  if (llvm::isa<EmptyOp>(op)) {
    return false;
  }
  return true;
}

/// Returns true for in-place ops with no tensor result that still need
/// input layout validation and upstream reshard propagation in the beam search.
inline bool isSinkOp(mlir::Operation *op) {
  return llvm::isa<FillCacheOp, PagedUpdateCacheOp>(op);
}

/// Returns true if this op should participate in the greedy beam search —
/// either because it produces a tensor output (normal path) or because it
/// is a sink that drives upstream input layout decisions.
inline bool isBeamSearchTarget(mlir::Operation *op) {
  return opHasTensorResult(op) || isSinkOp(op);
}

// Create affine maps that translate a virtual grid layout to a physical
// grid layout and vice versa for a single device based on the specified tensor
// memory layout.
//
// This function supports three types of tensor memory layouts:
// - WidthSharded: Maps a width-sharded virtual grid (1xN) to a physical grid
//   with the specified shape.
// - HeightSharded: Maps a height-sharded virtual grid (Mx1) to a physical grid
//   with the specified shape.
// - BlockSharded: Maps a block-sharded virtual grid (MxN) directly to a
//   physical grid with the specified shape.
//
// \param context The MLIR context.
// \param tensorMemoryLayout The tensor memory layout type.
// \param physicalGridShape The shape of the physical grid, defaults to {8, 8}.
//
// \return A pair of affine maps that translate the virtual grid layout to the
// physical grid layout and vice versa based on the specified tensor memory
// layout.
std::pair<AffineMap, AffineMap> createSingleDeviceVirtualToPhysicalAffineMaps(
    MLIRContext *context,
    const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
    const llvm::ArrayRef<int64_t> physicalGridShape = {8, 8});

// Returns unique op-specific attributes from a list of OpConfigs.
// Deduplicates by comparing OpConfig::OpSpecificAttrs values.
std::vector<mlir::tt::ttnn::OpConfig::OpSpecificAttrs>
getUniqueOpSpecificAttrs(const std::vector<mlir::tt::ttnn::OpConfig> &configs);

// Returns unique test configs for Matmul/Linear ops.
// Generates Cartesian product of unique (bufferType, memLayout) pairs
// with unique op-specific attrs, using ignorePhysicalLayout.
llvm::SmallVector<mlir::tt::ttnn::OpConfig> getUniqueTestConfigsForMatmulLinear(
    const std::vector<mlir::tt::ttnn::OpConfig> &consumerConfigs);

// Returns unique test configs for validation.
// - For non-Matmul/Linear ops: Only unique op-specific attrs (no output layout
//   needed).
// - For Matmul/Linear ops: Cartesian product of unique (bufferType, memLayout)
//   pairs with unique op-specific attrs, using ignorePhysicalLayout
llvm::SmallVector<mlir::tt::ttnn::OpConfig> getUniqueTestConfigs(
    const std::vector<mlir::tt::ttnn::OpConfig> &consumerConfigs,
    bool isMatmulOrLinear);

} // namespace mlir::tt::ttnn::optimizer_utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
