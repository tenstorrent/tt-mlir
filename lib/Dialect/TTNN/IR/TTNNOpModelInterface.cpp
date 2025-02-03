// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/Operation.h"

#include "llvm/Support/Error.h"
#include <cassert>
#include <cstddef>
#include <optional>
#include <tuple>

namespace mlir::tt::ttnn {

namespace detail {
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
checkDeviceWorkerGrid(mlir::Operation *op) {

  auto deviceAttr = mlir::tt::getCurrentScopeDevice(op);
  assert(deviceAttr);
  auto checkWorkerGrid =
      op_model::ttnn::Device::getDeviceConstraints(deviceAttr.getWorkerGrid());

  if (std::get<0>(checkWorkerGrid) == false) {
    return std::make_tuple(std::get<0>(checkWorkerGrid), std::nullopt,
                           std::get<1>(checkWorkerGrid));
  }

  return std::make_tuple(true, std::nullopt, std::nullopt);
}
} // namespace detail

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
ReluOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                         const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape =
      mlir::cast<RankedTensorType>(getDpsInputOperand(0)->get().getType())
          .getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResults().front().getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
  }

  return op_model::ttnn::ReluOpInterface::getOpConstraints(
      inputShape, inputs[0], outputShape, output);
}

llvm::Expected<size_t>
ReluOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                     const TTNNLayoutAttr &output) {

  assert(inputs.size() == 1);

  const auto inputShape =
      mlir::cast<RankedTensorType>(getDpsInputOperand(0)->get().getType())
          .getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResults().front().getType()).getShape();

  return op_model::ttnn::ReluOpInterface::getOpRuntime(inputShape, inputs[0],
                                                       outputShape, output);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
AddOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto inputShapeA =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto inputShapeB =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult(0).getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
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

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
SoftmaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape =
      mlir::cast<RankedTensorType>(getOperand().getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
  }

  return op_model::ttnn::SoftmaxOpInterface::getOpConstraints(
      inputShape, inputs[0], getDimension(), outputShape, output);
}

llvm::Expected<size_t>
SoftmaxOp::getOpRuntime(const std::vector<TTNNLayoutAttr> &inputs,
                        const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto inputShape =
      mlir::cast<RankedTensorType>(getOperand().getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  return op_model::ttnn::SoftmaxOpInterface::getOpRuntime(
      inputShape, inputs[0], getDimension(), outputShape, output);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
MatmulOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto inputShapeA =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto inputShapeB =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto outputShape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
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
