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

std::tuple<size_t, size_t, size_t>
ReluOp::getOpL1Usage(const std::vector<mlir::RankedTensorType> &input_rtts,
                     const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 1);

  const auto input_shape = input_rtts[0].getShape();
  const auto input_layout =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::ReluOpInterface::getOpL1Usage(
      input_shape, input_layout, output_shape, output_layout);
}

bool ReluOp::isOpLegal(const std::vector<mlir::RankedTensorType> &input_rtts,
                       const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 1);

  const auto input_shape = input_rtts[0].getShape();
  const auto input_layout =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::ReluOpInterface::isLegal(input_shape, input_layout,
                                                  output_shape, output_layout);
}

//===----------------------------------------------------------------------===//
// AddOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<size_t, size_t, size_t>
AddOp::getOpL1Usage(const std::vector<mlir::RankedTensorType> &input_rtts,
                    const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 2);

  const auto input_shape_a = input_rtts[0].getShape();
  const auto input_layout_a =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto input_shape_b = input_rtts[0].getShape();
  const auto input_layout_b =
      mlir::cast<TTNNLayoutAttr>(input_rtts[1].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::AddOpInterface::getOpL1Usage(
      input_shape_a, input_layout_a, input_shape_b, input_layout_b,
      output_shape, output_layout);
}

bool AddOp::isOpLegal(const std::vector<mlir::RankedTensorType> &input_rtts,
                      const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 2);

  const auto input_shape_a = input_rtts[0].getShape();
  const auto input_layout_a =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto input_shape_b = input_rtts[0].getShape();
  const auto input_layout_b =
      mlir::cast<TTNNLayoutAttr>(input_rtts[1].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::AddOpInterface::isLegal(input_shape_a, input_layout_a,
                                                 input_shape_b, input_layout_b,
                                                 output_shape, output_layout);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<size_t, size_t, size_t>
SoftmaxOp::getOpL1Usage(const std::vector<mlir::RankedTensorType> &input_rtts,
                        const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 1);

  const auto input_shape = input_rtts[0].getShape();
  const auto input_layout =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::SoftmaxOpInterface::getOpL1Usage(
      input_shape, input_layout, getDimension(), output_shape, output_layout);
}

bool SoftmaxOp::isOpLegal(const std::vector<mlir::RankedTensorType> &input_rtts,
                          const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 1);

  const auto input_shape = input_rtts[0].getShape();
  const auto input_layout =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::SoftmaxOpInterface::isLegal(
      input_shape, input_layout, getDimension(), output_shape, output_layout);
}

//===----------------------------------------------------------------------===//
// MatmulOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

std::tuple<size_t, size_t, size_t>
MatmulOp::getOpL1Usage(const std::vector<mlir::RankedTensorType> &input_rtts,
                       const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 2);

  const auto input_shape_a = input_rtts[0].getShape();
  const auto input_layout_a =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto input_shape_b = input_rtts[0].getShape();
  const auto input_layout_b =
      mlir::cast<TTNNLayoutAttr>(input_rtts[1].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::MatmulOpInterface::getOpL1Usage(
      input_shape_a, input_layout_a, input_shape_b, input_layout_b,
      output_shape, output_layout);
}

bool MatmulOp::isOpLegal(const std::vector<mlir::RankedTensorType> &input_rtts,
                         const mlir::RankedTensorType &output_rtt) {
  assert(input_rtts.size() == 2);

  const auto input_shape_a = input_rtts[0].getShape();
  const auto input_layout_a =
      mlir::cast<TTNNLayoutAttr>(input_rtts[0].getEncoding());
  const auto input_shape_b = input_rtts[0].getShape();
  const auto input_layout_b =
      mlir::cast<TTNNLayoutAttr>(input_rtts[1].getEncoding());
  const auto output_shape = output_rtt.getShape();
  const auto output_layout =
      mlir::cast<TTNNLayoutAttr>(output_rtt.getEncoding());

  return op_model::ttnn::MatmulOpInterface::isLegal(
      input_shape_a, input_layout_a, input_shape_b, input_layout_b,
      output_shape, output_layout);
}

} // namespace mlir::tt::ttnn
