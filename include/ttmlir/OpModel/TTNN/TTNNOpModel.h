// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/ArrayRef.h"

#include <tuple>

namespace mlir::tt::op_model::ttnn {

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

namespace Device {
std::tuple<bool, std::optional<std::string>>
getDeviceConstraints(const mlir::tt::GridAttr &workerGrid);
}; // namespace Device

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

namespace ReluOpInterface {
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace ReluOpInterface

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace AddOpInterface {
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                 llvm::ArrayRef<int64_t> inputShapeB,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
             llvm::ArrayRef<int64_t> inputShapeB,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace AddOpInterface

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

namespace SoftmaxOpInterface {
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace SoftmaxOpInterface

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

namespace MatmulOpInterface {
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                 llvm::ArrayRef<int64_t> inputShapeB,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
                 bool transposeB);

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
             llvm::ArrayRef<int64_t> inputShapeB,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
             bool transposeB);
}; // namespace MatmulOpInterface

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
