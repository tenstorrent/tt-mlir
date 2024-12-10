// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <llvm/ADT/ArrayRef.h>
#include <tuple>

namespace mlir::tt::op_model::ttnn {

//===----------------------------------------------------------------------===//
// Test MLIR<->METAL Conversion functionality
//===----------------------------------------------------------------------===//
namespace conversion {
void debug(const ::llvm::ArrayRef<int64_t> shape,
           const mlir::tt::ttnn::TTNNLayoutAttr &layout);
}; // namespace conversion

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

namespace ReluOpInterface {
bool isLegal(const llvm::ArrayRef<int64_t> &inputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const llvm::ArrayRef<int64_t> &inputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
}; // namespace ReluOpInterface

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace AddOpInterface {

bool isLegal(const llvm::ArrayRef<int64_t> &inputShape_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const llvm::ArrayRef<int64_t> &inputShape_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const llvm::ArrayRef<int64_t> &inputShape_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const llvm::ArrayRef<int64_t> &inputShape_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
}; // namespace AddOpInterface

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

namespace SoftmaxOpInterface {

bool isLegal(const llvm::ArrayRef<int64_t> &inputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const int dim_arg, const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const llvm::ArrayRef<int64_t> &inputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
             const int dim_arg, const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout);
}; // namespace SoftmaxOpInterface

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

namespace MatmulOpInterface {

bool isLegal(const llvm::ArrayRef<int64_t> &inputShape_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const llvm::ArrayRef<int64_t> &inputShape_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
             bool transpose_a = false, bool transpose_b = false);

std::tuple<size_t, size_t, size_t>
getOpL1Usage(const llvm::ArrayRef<int64_t> &inputShape_a,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
             const llvm::ArrayRef<int64_t> &inputShape_b,
             const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
             const llvm::ArrayRef<int64_t> &outputShape,
             const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
             bool transpose_a = false, bool transpose_b = false);
}; // namespace MatmulOpInterface

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
