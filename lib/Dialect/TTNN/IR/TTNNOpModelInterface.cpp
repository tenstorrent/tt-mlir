// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/Operation.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include <cassert>
#include <tuple>

namespace mlir::tt::ttnn {

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

  auto deviceAttr = mlir::tt::getCurrentScopeDevice(getOperation());
  assert(deviceAttr);
  auto workerGrid = deviceAttr.getWorkerGrid();

  return op_model::ttnn::ReluOpInterface::getOpConstraints(
      input_shape, inputs[0], output_shape, output, workerGrid);
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

  auto deviceAttr = mlir::tt::getCurrentScopeDevice(getOperation());
  assert(deviceAttr);
  auto workerGrid = deviceAttr.getWorkerGrid();

  return op_model::ttnn::AddOpInterface::getOpConstraints(
      input_shape_a, inputs[0], input_shape_b, inputs[1], output_shape, output,
      workerGrid);
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

  auto deviceAttr = mlir::tt::getCurrentScopeDevice(getOperation());
  assert(deviceAttr);
  auto workerGrid = deviceAttr.getWorkerGrid();

  return op_model::ttnn::SoftmaxOpInterface::getOpConstraints(
      input_shape, inputs[0], getDimension(), output_shape, output, workerGrid);
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

  auto deviceAttr = mlir::tt::getCurrentScopeDevice(getOperation());
  assert(deviceAttr);
  auto workerGrid = deviceAttr.getWorkerGrid();

  return op_model::ttnn::MatmulOpInterface::getOpConstraints(
      input_shape_a, inputs[0], input_shape_b, inputs[1], output_shape, output,
      false, false, workerGrid);
}

} // namespace mlir::tt::ttnn
