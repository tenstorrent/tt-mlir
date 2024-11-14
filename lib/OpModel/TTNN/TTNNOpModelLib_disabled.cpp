// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.hpp"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::op_model::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

bool ReluOpInterface::isLegal(const mlir::tt::LayoutAttr &inputLayout,
                              const mlir::tt::LayoutAttr &outputLayout) {
  return true;
}

std::tuple<size_t, size_t, size_t>
ReluOpInterface::getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                              const mlir::tt::LayoutAttr &outputLayout) {
  return std::make_tuple(0, 0, 0);
}

} // namespace mlir::tt::op_model::ttnn