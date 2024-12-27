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

  const auto input_shape =
      mlir::cast<RankedTensorType>(getDpsInputOperand(0)->get().getType())
          .getShape();

  const auto output_shape =
      mlir::cast<RankedTensorType>(getResults().front().getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
  }

  return op_model::ttnn::ReluOpInterface::getOpConstraints(
      input_shape, inputs[0], output_shape, output);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
AddOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                        const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto input_shape_a =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto input_shape_b =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto output_shape =
      mlir::cast<RankedTensorType>(getResult(0).getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
  }

  return op_model::ttnn::AddOpInterface::getOpConstraints(
      input_shape_a, inputs[0], input_shape_b, inputs[1], output_shape, output);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
SoftmaxOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                            const TTNNLayoutAttr &output) {
  assert(inputs.size() == 1);

  const auto input_shape =
      mlir::cast<RankedTensorType>(getOperand().getType()).getShape();

  const auto output_shape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
  }

  return op_model::ttnn::SoftmaxOpInterface::getOpConstraints(
      input_shape, inputs[0], getDimension(), output_shape, output);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
MatmulOp::getOpConstraints(const std::vector<TTNNLayoutAttr> &inputs,
                           const TTNNLayoutAttr &output) {
  assert(inputs.size() == 2);

  const auto input_shape_a =
      mlir::cast<RankedTensorType>(getOperand(0).getType()).getShape();
  const auto input_shape_b =
      mlir::cast<RankedTensorType>(getOperand(1).getType()).getShape();

  const auto output_shape =
      mlir::cast<RankedTensorType>(getResult().getType()).getShape();

  auto check = detail::checkDeviceWorkerGrid(getOperation());
  if (std::get<bool>(check) == false) {
    return check;
  }

  return op_model::ttnn::MatmulOpInterface::getOpConstraints(
      input_shape_a, inputs[0], input_shape_b, inputs[1], output_shape, output,
      false, false);
}

} // namespace mlir::tt::ttnn
