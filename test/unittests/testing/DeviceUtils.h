// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_TESTING_DEVICE_UTILS_H
#define TT_TESTING_DEVICE_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

// Use a dedicated namespace to avoid shadowing gtest's `::testing` from
// translation units that nest tests inside `namespace mlir::tt::...`.
namespace mlir::tt::test_utils {

// Build a fake `ttcore::DeviceAttr` with the given worker-grid shape, suitable
// for unit tests that need to feed a `DeviceAttr` into TTNN APIs (e.g.
// `TTNNLayoutAttr::Builder::buildWithCanonicalCorePlacement`). Only the
// worker grid is meaningful; affine maps and other fields are populated with
// minimal placeholders.
inline mlir::tt::ttcore::DeviceAttr
getFakeDeviceAttr(mlir::MLIRContext *ctx,
                  llvm::ArrayRef<int64_t> workerGridShape = {8, 8}) {
  auto deviceIdx = mlir::getAffineConstantExpr(0, ctx);
  auto shardOffset = mlir::getAffineConstantExpr(0, ctx);
  auto d0 = mlir::getAffineDimExpr(0, ctx);
  auto d1 = mlir::getAffineDimExpr(1, ctx);
  auto d2 = mlir::getAffineDimExpr(2, ctx);
  auto virtToPhysicalMap = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1}, ctx);
  auto physicalToVirtMap =
      mlir::AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, {d1, d2}, ctx);
  auto map4 = mlir::AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1, shardOffset}, ctx);
  auto workerGrid = mlir::tt::ttcore::GridAttr::get(
      ctx, workerGridShape, virtToPhysicalMap, physicalToVirtMap);
  auto dramGrid = mlir::tt::ttcore::GridAttr::get(ctx, {1, 1});
  return mlir::tt::ttcore::DeviceAttr::get(ctx, workerGrid, dramGrid, map4,
                                           map4, {1}, {0}, {});
}

} // namespace mlir::tt::test_utils

#endif // TT_TESTING_DEVICE_UTILS_H
