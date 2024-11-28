// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <tuple>

namespace mlir::tt::op_model::ttnn {

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

namespace ReluOpInterface {
bool isLegal(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
}; // namespace ReluOpInterface

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace AddOpInterface {

bool isLegal(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
}; // namespace AddOpInterface

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

namespace SoftmaxOpInterface {

bool isLegal(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const int dim_arg,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const int dim_arg,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
}; // namespace SoftmaxOpInterface

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

namespace MatmulOpInterface {

bool isLegal(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
             bool transpose_a = false, bool transpose_b = false);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
             bool transpose_a = false, bool transpose_b = false);
}; // namespace MatmulOpInterface

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
