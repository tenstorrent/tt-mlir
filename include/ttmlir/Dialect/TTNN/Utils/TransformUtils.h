// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_TRANSFORMUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_TRANSFORMUTILS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::utils {
// Get or insert device for the given operation.
inline GetDeviceOp
getOrInsertDevice(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                  llvm::function_ref<mlir::Location(mlir::Location)> locFn =
                      ::ttmlir::utils::identity<mlir::Location>) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp;
    }
  }

  DeviceAttr deviceAttr = getCurrentScopeDevice(op);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  llvm::SmallVector<int64_t> meshShape{deviceAttr.getMeshShape()};
  if (meshShape.empty()) {
    meshShape = llvm::SmallVector<int64_t, 2>{1, 1};
  }
  auto deviceOp = rewriter.create<ttnn::GetDeviceOp>(
      locFn(op->getLoc()), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), meshShape[0], meshShape[1]));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp;
}

// Helper method to insert a ToLayoutOp to convert the input operand to the
// desired tensor layout, buffer type and memory layout.
ToLayoutOp
createToLayoutOp(mlir::Operation *op,
                 mlir::TypedValue<RankedTensorType> inputValue,
                 PatternRewriter &rewriter, Layout targetTensorLayout,
                 BufferType targetTensorBufferType,
                 std::optional<TensorMemoryLayout> targetTensorMemoryLayout,
                 DataType targetTensorDataType);
} // namespace mlir::tt::ttnn::utils

#endif
