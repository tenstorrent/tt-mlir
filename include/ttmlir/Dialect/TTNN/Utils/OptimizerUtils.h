// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

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

} // namespace mlir::tt::ttnn::optimizer_utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
