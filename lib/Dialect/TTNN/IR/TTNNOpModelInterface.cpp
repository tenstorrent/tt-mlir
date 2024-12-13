// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

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
ReluOp::getOpConstraints(
    const std::vector<std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr>> &inputs,
    const std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr> &output) {
  assert(inputs.size() == 1);

  const auto &input_shape = std::get<0>(inputs[0]);
  const auto &input_layout = std::get<1>(inputs[0]);
  const auto &output_shape = std::get<0>(output);
  const auto &output_layout = std::get<1>(output);

  return op_model::ttnn::ReluOpInterface::getOpConstraints(
      input_shape, input_layout, output_shape, output_layout);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
AddOp::getOpConstraints(
    const std::vector<std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr>> &inputs,
    const std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr> &output) {
  assert(inputs.size() == 2);

  const auto &input_shape_a = std::get<0>(inputs[0]);
  const auto &input_layout_a = std::get<1>(inputs[0]);
  const auto &input_shape_b = std::get<0>(inputs[1]);
  const auto &input_layout_b = std::get<1>(inputs[1]);
  const auto &output_shape = std::get<0>(output);
  const auto &output_layout = std::get<1>(output);

  return op_model::ttnn::AddOpInterface::getOpConstraints(
      input_shape_a, input_layout_a, input_shape_b, input_layout_b,
      output_shape, output_layout);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
SoftmaxOp::getOpConstraints(
    const std::vector<std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr>> &inputs,
    const std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr> &output) {
  assert(inputs.size() == 1);

  const auto &input_shape = std::get<0>(inputs[0]);
  const auto &input_layout = std::get<1>(inputs[0]);
  const auto &output_shape = std::get<0>(output);
  const auto &output_layout = std::get<1>(output);

  return op_model::ttnn::SoftmaxOpInterface::getOpConstraints(
      input_shape, input_layout, getDimension(), output_shape, output_layout);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
MatmulOp::getOpConstraints(
    const std::vector<std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr>> &inputs,
    const std::tuple<ArrayRef<int64_t>, TTNNLayoutAttr> &output) {
  assert(inputs.size() == 2);

  const auto &input_shape_a = std::get<0>(inputs[0]);
  const auto &input_layout_a = std::get<1>(inputs[0]);
  const auto &input_shape_b = std::get<0>(inputs[1]);
  const auto &input_layout_b = std::get<1>(inputs[1]);
  const auto &output_shape = std::get<0>(output);
  const auto &output_layout = std::get<1>(output);

  return op_model::ttnn::MatmulOpInterface::getOpConstraints(
      input_shape_a, input_layout_a, input_shape_b, input_layout_b,
      output_shape, output_layout);
}

} // namespace mlir::tt::ttnn
