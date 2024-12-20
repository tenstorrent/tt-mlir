// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_VIRTUALTOPHYSICALGRID_H
#define TTMLIR_DIALECT_TTNN_UTILS_VIRTUALTOPHYSICALGRID_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"

namespace mlir::tt::ttnn::utils {
auto SingleDeviceCreateVirtualToPhysicalLayoutMap(
    MLIRContext *context,
    const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
    const llvm::ArrayRef<int64_t> physicalGridShape = {8, 8}) {

  auto workerDeviceIdx = mlir::getAffineConstantExpr(0, context);

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded: {
    // create affine map that maps width sharded virtual grid 1xN to the
    // physical grid gridShape[0] x gridShape[1]
    auto virtualWidth = mlir::getAffineDimExpr(1, context); // d1
    auto workerCoreW =
        mlir::getAffineConstantExpr(physicalGridShape[1], context);
    auto widthMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualWidth.floorDiv(workerCoreW),
         virtualWidth % workerCoreW},
        context);
    return widthMap;
  }
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded: {
    // create affine map that maps height sharded virtual grid Mx1 to the
    // physical grid gridShape[0] x gridShape[1]
    auto virtualHeight = mlir::getAffineDimExpr(0, context); // d0
    auto workerCoreH =
        mlir::getAffineConstantExpr(physicalGridShape[0], context);
    auto heightMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0,
        {workerDeviceIdx, virtualHeight.floorDiv(workerCoreH),
         virtualHeight % workerCoreH},
        context);
    return heightMap;
  }
  default:
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded: {
    auto d0 = mlir::getAffineDimExpr(0, context); // d0
    auto d1 = mlir::getAffineDimExpr(1, context); // d1
    auto blockMap = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {workerDeviceIdx, d0, d1}, context);
    blockMap.dump();
    return blockMap;
  }
  }
}
} // namespace mlir::tt::ttnn::utils
#endif // TTMLIR_DIALECT_TTNN_UTILS_VIRTUALTOPHYSICALGRID_H