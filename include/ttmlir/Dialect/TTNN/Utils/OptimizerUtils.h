// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace mlir::tt::ttnn::optimizer_utils {

// Creates an affine map that translates a virtual grid layout to a physical
// grid layout for a single device based on the specified tensor memory layout.
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
// \return An affine map that translates the virtual grid layout to the
// physical grid layout based on the specified tensor memory layout.
AffineMap createSingleDeviceVirtualToPhysicalAffineMap(
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

// Validate that an op can accept a specific input layout and produce its
// expected output layout. For matmul/linear ops, uses withIgnorePhysicalLayout
// during validation. Returns true only if validation succeeds AND the actual
// output layout matches the config's expected output layout.
bool validateOpWithInputLayout(Operation *op, size_t inputOperandIndex,
                               TTNNLayoutAttr inputLayout,
                               const mlir::tt::ttnn::OpConfig &config);

} // namespace mlir::tt::ttnn::optimizer_utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
