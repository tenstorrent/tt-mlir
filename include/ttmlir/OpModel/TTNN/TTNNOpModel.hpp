// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt::op_model::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

namespace ReluOpInterface {
bool isLegal(const mlir::tt::LayoutAttr &inputLayout,
             const mlir::tt::LayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
             const mlir::tt::LayoutAttr &outputLayout);
}; // namespace ReluOpInterface

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace AddOpInterface {

bool isLegal(const mlir::tt::LayoutAttr &inputLayout_a,
             const mlir::tt::LayoutAttr &inputLayout_b,
             const mlir::tt::LayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout_a,
             const mlir::tt::LayoutAttr &inputLayout_b,
             const mlir::tt::LayoutAttr &outputLayout);
}; // namespace AddOpInterface

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

namespace SoftmaxOpInterface {

bool isLegal(const mlir::tt::LayoutAttr &inputLayout, const int dim_arg,
             const mlir::tt::LayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout, const int dim_arg,
             const mlir::tt::LayoutAttr &outputLayout);
}; // namespace SoftmaxOpInterface

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

namespace MatmulOpInterface {

bool isLegal(const mlir::tt::LayoutAttr &inputLayout_a,
             const mlir::tt::LayoutAttr &inputLayout_b,
             const mlir::tt::LayoutAttr &outputLayout, bool transpose_a = false,
             bool transpose_b = false);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout_a,
             const mlir::tt::LayoutAttr &inputLayout_b,
             const mlir::tt::LayoutAttr &outputLayout, bool transpose_a = false,
             bool transpose_b = false);
}; // namespace MatmulOpInterface

} // namespace mlir::tt::op_model::ttnn
