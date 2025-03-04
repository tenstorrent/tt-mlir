// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/Operation.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir::tt::ttnn {

namespace detail {
llvm::Expected<bool> checkDeviceWorkerGrid(mlir::Operation *op) {
  auto deviceAttr = mlir::tt::getCurrentScopeDevice(op);
  assert(deviceAttr);
  return op_model::ttnn::Device::getDeviceConstraints(
      deviceAttr.getWorkerGrid());
}

std::optional<llvm::SmallVector<int64_t>>
convertReductionArg(std::optional<mlir::ArrayAttr> arrayOpt) {
  if (!arrayOpt.has_value()) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t> reduceDims;

  for (const mlir::Attribute &reduceDim : *arrayOpt) {
    reduceDims.push_back(mlir::cast<mlir::IntegerAttr>(reduceDim).getInt());
  }

  return reduceDims;
}

} // namespace detail

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
ReluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResults().front().getType()).getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::ReluOpInterface::getOpConstraints(
      inputShape, inputs[0], outputShape, output);
}

llvm::Expected<size_t>
ReluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const TTNNLayoutAttr &output) {

  assert(inputs.size() == 1);

  const auto inputShape =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResults().front().getType()).getShape();

  return op_model::ttnn::ReluOpInterface::getOpRuntime(inputShape, inputs[0],
                                                       outputShape, output);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
AddOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto inputShapeA =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto inputShapeB =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult(0).getType()).getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::AddOpInterface::getOpConstraints(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape, output);
}

llvm::Expected<size_t>
AddOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                    const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto inputShapeA =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto inputShapeB =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult(0).getType()).getShape();

  return op_model::ttnn::AddOpInterface::getOpRuntime(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape, output);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
SoftmaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::SoftmaxOpInterface::getOpConstraints(
      inputShape, inputs[0], getDimension(), outputShape, output);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  return op_model::ttnn::SoftmaxOpInterface::getOpRuntime(
      inputShape, inputs[0], getDimension(), outputShape, output);
}

//===----------------------------------------------------------------------===//
// MeanOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
MeanOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::MeanOpInterface::getOpConstraints(
      inputShape, inputs[0], detail::convertReductionArg(getDimArg()),
      getKeepDim(), output);
}

llvm::Expected<size_t>
MeanOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return op_model::ttnn::MeanOpInterface::getOpRuntime(
      inputShape, inputs[0], detail::convertReductionArg(getDimArg()),
      getKeepDim(), output);
}

//===----------------------------------------------------------------------===//
// ReshapeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
ReshapeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::ReshapeOpInterface::getOpConstraints(
      inputShape, inputs[0], outputShape, output);
}

llvm::Expected<size_t>
ReshapeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();
  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  return op_model::ttnn::ReshapeOpInterface::getOpRuntime(inputShape, inputs[0],
                                                          outputShape, output);
}

//===----------------------------------------------------------------------===//
// TransposeOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
TransposeOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                              const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::TransposeOpInterface::getOpConstraints(
      inputShape, inputs[0], getDim0(), getDim1(), output);
}

llvm::Expected<size_t>
TransposeOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                          const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape = getInput().getType().getShape();

  return op_model::ttnn::TransposeOpInterface::getOpRuntime(
      inputShape, inputs[0], getDim0(), getDim1(), output);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

llvm::Expected<std::tuple<size_t, size_t, size_t>>
MatmulOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto inputShapeA =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto inputShapeB =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  llvm::Expected<bool> check = detail::checkDeviceWorkerGrid(getOperation());
  if (!check) {
    return check.takeError();
  }

  return op_model::ttnn::MatmulOpInterface::getOpConstraints(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape, output,
      false, false);
}

llvm::Expected<size_t>
MatmulOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                       const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto inputShapeA =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto inputShapeB =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  return op_model::ttnn::MatmulOpInterface::getOpRuntime(
      inputShapeA, inputs[0], inputShapeB, inputs[1], outputShape, output,
      false, false);
}

} // namespace mlir::tt::ttnn
