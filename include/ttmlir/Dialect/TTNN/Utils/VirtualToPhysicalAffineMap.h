// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_VIRTUALTOPHYSICALGRID_H
#define TTMLIR_DIALECT_TTNN_UTILS_VIRTUALTOPHYSICALGRID_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"

namespace mlir::tt::ttnn::utils {

/// Creates an affine map that translates a virtual grid layout to a physical
/// grid layout for a single device based on the specified tensor memory layout.
///
/// This function supports three types of tensor memory layouts:
/// - WidthSharded: Maps a width-sharded virtual grid (1xN) to a physical grid
///   with the specified shape.
/// - HeightSharded: Maps a height-sharded virtual grid (Mx1) to a physical grid
///   with the specified shape.
/// - BlockSharded: Maps a block-sharded virtual grid (MxN) directly to a
///   physical grid with the specified shape.
///
/// \param context The MLIR context.
/// \param tensorMemoryLayout The tensor memory layout type.
/// \param physicalGridShape The shape of the physical grid, defaults to {8, 8}.
///
/// \return An affine map that translates the virtual grid layout to the
/// physical grid layout based on the specified tensor memory layout.
AffineMap CreateSingleDeviceVirtualToPhysicalAffineMap(
    MLIRContext *context,
    const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
    const llvm::ArrayRef<int64_t> physicalGridShape = {8, 8}) {

  AffineExpr workerDeviceIdx = mlir::getAffineConstantExpr(0, context);

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded: {
    // create affine map that maps width sharded virtual grid 1xN to the
    // physical grid gridShape[0] x gridShape[1]
    AffineExpr virtualWidth = mlir::getAffineDimExpr(1, context); // d1
    AffineExpr workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    AffineMap widthMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualWidth.floorDiv(workerCoreW),
         virtualWidth % workerCoreW},
        context);
    return widthMap;
  }
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded: {
    // create affine map that maps height sharded virtual grid Mx1 to the
    // physical grid gridShape[0] x gridShape[1]
    AffineExpr virtualHeight = mlir::getAffineDimExpr(0, context); // d0
    AffineExpr workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    AffineMap heightMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualHeight.floorDiv(workerCoreW),
         virtualHeight % workerCoreW},
        context);
    return heightMap;
  }
  default:
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded: {
    AffineExpr d0 = mlir::getAffineDimExpr(0, context); // d0
    AffineExpr d1 = mlir::getAffineDimExpr(1, context); // d1
    AffineMap blockMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {workerDeviceIdx, d0, d1}, context);
    return blockMap;
  }
  }
}
} // namespace mlir::tt::ttnn::utils
#endif // TTMLIR_DIALECT_TTNN_UTILS_VIRTUALTOPHYSICALGRID_H
