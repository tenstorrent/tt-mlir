// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.hpp"

#include <cassert>
#include <tuple>

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

size_t ReluOp::getOpPerfCycles(const std::vector<tt::LayoutAttr> &input_layouts,
                               const tt::LayoutAttr &output_layout) {
  // TODO(mbezulj) wire to tt-metal once we have API
  return 5;
}

std::tuple<size_t, size_t, size_t>
ReluOp::getOpL1Usage(const std::vector<tt::LayoutAttr> &input_layouts,
                     const tt::LayoutAttr &output_layout) {
  assert(input_layouts.size() == 1);
  return op_model::ttnn::ReluOpInterface::getOpL1Usage(input_layouts[0],
                                                       output_layout);
}

bool ReluOp::isOpLegal(const std::vector<tt::LayoutAttr> &input_layouts,
                       const tt::LayoutAttr &output_layout) {
  assert(input_layouts.size() == 1);
  return op_model::ttnn::ReluOpInterface::isLegal(input_layouts[0],
                                                  output_layout);
}

} // namespace mlir::tt::ttnn
