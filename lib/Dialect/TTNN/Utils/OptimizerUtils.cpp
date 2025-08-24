// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <unordered_set>

namespace mlir::tt::ttnn::optimizer_utils {

AffineMap createSingleDeviceVirtualToPhysicalAffineMap(
    MLIRContext *context,
    const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
    const llvm::ArrayRef<int64_t> physicalGridShape) {

  AffineExpr workerDeviceIdx = mlir::getAffineConstantExpr(0, context);

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded: {
    // Create affine map that maps width sharded virtual grid 1xN to the
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
    // Create affine map that maps height sharded virtual grid Mx1 to the
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

} // namespace mlir::tt::ttnn::optimizer_utils
