// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

// #include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::op_model::ttnn {

struct ReluOpInterface {
  static bool isLegal(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
                      const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

  static std::tuple<size_t, size_t, size_t>
  getOpL1Usage(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
               const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
};

} // namespace mlir::tt::op_model::ttnn
