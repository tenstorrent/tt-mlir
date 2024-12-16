// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn::utils {
// Gets or inserts a GetDeviceOp at the top of the current block of the given
// operation.
Value getOrInsertDevice(PatternRewriter &rewriter, Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp.getResult();
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
      op->getLoc(), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), meshShape[0], meshShape[1]));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp.getResult();
}
} // namespace mlir::tt::ttnn::utils
