// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"

namespace mlir {
namespace tt {
namespace ttir_to_ttnn::utils {

Value getOrInsertDevice(ConversionPatternRewriter &rewriter, Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp.getResult();
    }
  }

  DeviceAttr deviceAttr = getCurrentScopeDevice(op);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  auto deviceOp = rewriter.create<ttnn::GetDeviceOp>(
      op->getLoc(), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), 1, 1));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp.getResult();
}

ttnn::ReshapeOp generateReshape(Value input, ArrayRef<int64_t> newShape,
                                PatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto outputType = inputType.cloneWith(newShape, inputType.getElementType());

  std::vector<int32_t> newShapeI32(newShape.begin(), newShape.end());
  return rewriter.create<ttnn::ReshapeOp>(
      input.getLoc(), outputType, input, rewriter.getI32ArrayAttr(newShapeI32));
}

ttnn::ReshapeOp generateNHWFlatten(Value input, PatternRewriter &rewriter) {
  std::vector<int64_t> shape =
      mlir::cast<RankedTensorType>(input.getType()).getShape().vec();

  assert(shape.size() == 4 && "Must have 4-dim tensor as conv2d input");

  std::vector<int64_t> newShape = {1, 1, shape[0] * shape[1] * shape[2],
                                   shape[3]};
  return generateReshape(input, newShape, rewriter);
}

} // namespace ttir_to_ttnn::utils
} // namespace tt
} // namespace mlir
