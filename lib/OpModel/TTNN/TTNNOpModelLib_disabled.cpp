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

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

bool AddOpInterface::isLegal(const mlir::tt::LayoutAttr &inputLayout_a,
                             const mlir::tt::LayoutAttr &inputLayout_b,
                             const mlir::tt::LayoutAttr &outputLayout) {
  return true;
}

std::tuple<size_t, size_t, size_t>
AddOpInterface::getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                             const mlir::tt::LayoutAttr &inputLayout_b,
                             const mlir::tt::LayoutAttr &outputLayout) {
  return std::make_tuple(0, 0, 0);
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

bool SoftmaxOpInterface::isLegal(const mlir::tt::LayoutAttr &inputLayout,
                                 const int dim_arg,
                                 const mlir::tt::LayoutAttr &outputLayout) {
  return true;
}

std::tuple<size_t, size_t, size_t>
SoftmaxOpInterface::getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                                 const int dim_arg,
                                 const mlir::tt::LayoutAttr &outputLayout) {
  return std::make_tuple(0, 0, 0);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

bool MatmulOpInterface::isLegal(const mlir::tt::LayoutAttr &inputLayout_a,
                                const mlir::tt::LayoutAttr &inputLayout_b,
                                const mlir::tt::LayoutAttr &outputLayout,
                                bool transpose_a, bool transpose_b) {
  return true;
}

std::tuple<size_t, size_t, size_t>
MatmulOpInterface::getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout_a,
                                const mlir::tt::LayoutAttr &inputLayout_b,
                                const mlir::tt::LayoutAttr &outputLayout,
                                bool transpose_a, bool transpose_b) {
  return std::make_tuple(0, 0, 0);
}

} // namespace mlir::tt::op_model::ttnn
