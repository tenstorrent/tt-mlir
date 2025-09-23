// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::utils {
static GetDeviceOp insertGetDeviceOp(RewriterBase &rewriter,
                                     ttcore::DeviceAttr deviceAttr,
                                     Location loc) {
  llvm::SmallVector<int64_t> meshShape{deviceAttr.getMeshShape()};
  if (meshShape.empty()) {
    meshShape = llvm::SmallVector<int64_t, 2>{1, 1};
  }
  // TODO (jnie): Currently hardcoding the mesh offset to 0x0
  // Need a proper plan to dynamically determine this.
  llvm::SmallVector<int64_t, 2> meshOffset{0, 0};
  return rewriter.create<ttnn::GetDeviceOp>(
      loc, rewriter.getType<DeviceType>(),
      ttnn::MeshShapeAttr::get(rewriter.getContext(), meshShape[0],
                               meshShape[1]),
      ttnn::MeshOffsetAttr::get(rewriter.getContext(), meshOffset[0],
                                meshOffset[1]));
}

// Gets or inserts a GetDeviceOp at the top of the current block of the given
// operation.
GetDeviceOp getOrInsertDevice(RewriterBase &rewriter, Operation *op) {
  Block *block = op->getBlock();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp;
    }
  }

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  GetDeviceOp deviceOp = insertGetDeviceOp(rewriter, deviceAttr, op->getLoc());
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp;
}

GetDeviceOp getOrInsertDevice(RewriterBase &rewriter, Block *block) {
  mlir::Operation *parentOp = block->getParentOp();
  for (auto &op : block->getOperations()) {
    if (auto deviceOp = dyn_cast<ttnn::GetDeviceOp>(op)) {
      return deviceOp;
    }
  }

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(parentOp);
  auto currentInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(block, block->begin());
  GetDeviceOp deviceOp =
      insertGetDeviceOp(rewriter, deviceAttr, parentOp->getLoc());
  rewriter.restoreInsertionPoint(currentInsertionPoint);
  return deviceOp;
}

// Helper method to insert a ToLayoutOp to convert the input operand to the
// desired tensor layout, buffer type and memory layout.
ToLayoutOp createToLayoutOp(Operation *op,
                            mlir::TypedValue<RankedTensorType> inputValue,
                            RewriterBase &rewriter, Layout targetTensorLayout,
                            BufferType targetTensorBufferType,
                            TensorMemoryLayoutAttr targetTensorMemoryLayout,
                            ttcore::DataType targetTensorDataType,
                            llvm::StringRef locSuffix) {
  TTNNLayoutAttr inputLayoutAttr =
      getLayoutAttrFromTensor(inputValue.getType());

  // Create element type based on tensor layout.
  Type elementType = getElementType(rewriter.getContext(), targetTensorLayout,
                                    targetTensorDataType);

  // Get the input operand type.
  RankedTensorType inputToLayoutOpType = inputValue.getType();

  // Create the new encoding for the output tensor type.
  TTNNLayoutAttr toLayoutOpResultEncoding =
      inputLayoutAttr
          .withElementType(elementType, inputToLayoutOpType.getShape())
          .withBufferType(targetTensorBufferType)
          .withMemoryLayout(targetTensorMemoryLayout);

  // Create the output result type with the new data type and encoding.
  RankedTensorType toLayoutOpResultType = RankedTensorTypeFactory::create(
      RankedTensorTypeFactory::create(
          inputToLayoutOpType,
          mlir::tt::ttcore::dataTypeToElementType(rewriter.getContext(),
                                                  targetTensorDataType)),
      toLayoutOpResultEncoding);

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);

  // Create the output memory config attribute.
  ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
      rewriter.getContext(), targetTensorMemoryLayout,
      ttnn::BufferTypeAttr::get(rewriter.getContext(), targetTensorBufferType),
      utils::createShardSpecIfNeeded(
          mlir::cast<TTNNLayoutAttr>(toLayoutOpResultType.getEncoding()),
          deviceAttr.getWorkerGrid()));

  Location loc = ttmlir::utils::appendLocationSuffix(op->getLoc(), locSuffix);
  // Create a ToLayoutOp to convert the input operand to the desired
  // tensor layout, buffer type and memory layout.
  return rewriter.create<ttnn::ToLayoutOp>(
      loc, toLayoutOpResultType, inputValue,
      LayoutAttr::get(rewriter.getContext(), targetTensorLayout),
      ttcore::DataTypeAttr::get(rewriter.getContext(), targetTensorDataType),
      outputMemConfigAttr);
}

} // namespace mlir::tt::ttnn::utils
