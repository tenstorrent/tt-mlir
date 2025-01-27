// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::utils {
// Gets or inserts a GetDeviceOp at the top of the current block of the given
// operation.
GetDeviceOp getOrInsertDevice(PatternRewriter &rewriter, Operation *op) {
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
      op->getLoc(), rewriter.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(op->getContext(), meshShape[0], meshShape[1]));
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp;
}

// Helper method to insert a ToLayoutOp to convert the input operand to the
// desired tensor layout, buffer type and memory layout.
ToLayoutOp
createToLayoutOp(Operation *op, mlir::TypedValue<RankedTensorType> inputValue,
                 PatternRewriter &rewriter, Layout targetTensorLayout,
                 BufferType targetTensorBufferType,
                 std::optional<TensorMemoryLayout> targetTensorMemoryLayout,
                 DataType targetTensorDataType) {
  TTNNLayoutAttr inputLayoutAttr =
      getLayoutAttrFromTensor(inputValue.getType());

  // Create element type based on tensor layout.
  Type elementType = getElementType(rewriter.getContext(), targetTensorLayout,
                                    targetTensorDataType);

  // Create tensor memory layout attribute.
  ttnn::TensorMemoryLayoutAttr outputMemLayoutAttr =
      targetTensorMemoryLayout.has_value()
          ? ttnn::TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                              targetTensorMemoryLayout.value())
          : nullptr;

  // Get the input operand type.
  RankedTensorType inputToLayoutOpType = inputValue.getType();

  // Create the new encoding for the output tensor type.
  TTNNLayoutAttr toLayoutOpResultEncoding =
      inputLayoutAttr
          .withElementType(rewriter.getContext(), elementType,
                           inputToLayoutOpType.getShape())
          .withBufferType(rewriter.getContext(), targetTensorBufferType)
          .withMemoryLayout(rewriter.getContext(), outputMemLayoutAttr);

  // Create the output result type with the new data type and encoding.
  RankedTensorType toLayoutOpResultType =
      ttnn::utils::createRankedTensorTypeWithEncoding(
          ttnn::utils::createRankedTensorTypeWithElementType(
              inputToLayoutOpType,
              utils::createRowMajorTypeFromDtype(rewriter.getContext(),
                                                 targetTensorDataType)),
          toLayoutOpResultEncoding);

  // Create the output memory config attribute.
  ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
      rewriter.getContext(),
      ttnn::BufferTypeAttr::get(rewriter.getContext(), targetTensorBufferType),
      ttnn::ShardSpecAttr::get(
          op->getContext(),
          ttnn::ShapeAttr::get(rewriter.getContext(),
                               toLayoutOpResultEncoding.getShardShape())),
      outputMemLayoutAttr);

  // Get the device value if the tensor output is not on the host.
  auto deviceValue = targetTensorBufferType == ttnn::BufferType::SystemMemory
                         ? nullptr
                         : Value(utils::getOrInsertDevice(rewriter, op));

  // Create a ToLayoutOp to convert the input operand to the desired
  // tensor layout, buffer type and memory layout.
  return rewriter.create<ttnn::ToLayoutOp>(
      op->getLoc(), toLayoutOpResultType, inputValue,
      LayoutAttr::get(rewriter.getContext(), targetTensorLayout),
      DataTypeAttr::get(rewriter.getContext(), targetTensorDataType),
      outputMemConfigAttr, deviceValue);
}
} // namespace mlir::tt::ttnn::utils
