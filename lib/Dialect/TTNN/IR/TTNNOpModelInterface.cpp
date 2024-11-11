// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.cpp.inc"
#include <tuple>

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp - TTNN Op Model Interface
//===----------------------------------------------------------------------===//

size_t ReluOp::getOpPerfCycles(const std::vector<TensorConfigAttr> &input_layouts,
                               const tt::LayoutAttr &output_layout) {
  // TODO(mbezulj) wire to tt-metal once we have API
  return 5;
}

std::tuple<size_t, size_t, size_t>
ReluOp::getOpL1Usage(const std::vector<TensorConfigAttr> &input_layouts,
                     const tt::LayoutAttr &output_layout) {
  // TODO(mbezulj) wire to tt-metal once we have API
  return std::make_tuple(1024, 2048, 1024);
}

bool ReluOp::isOpLegal(const std::vector<TensorConfigAttr> &input_layouts,
                       const tt::LayoutAttr &output_layout) {
  // TODO(mbezulj) wire to tt-metal once we have API
  return true;
}

} // namespace mlir::tt::ttnn
