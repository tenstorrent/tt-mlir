// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/ArrayRef.h"

#include <tuple>

namespace mlir::tt::op_model::ttnn {

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

namespace Device {
llvm::Expected<bool> getDeviceConstraints(mlir::tt::GridAttr workerGrid);
}; // namespace Device

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

namespace ReluOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace ReluOpInterface

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace AddOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                 llvm::ArrayRef<int64_t> inputShapeB,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
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
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace SoftmaxOpInterface

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

namespace MeanOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace MeanOpInterface

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

namespace ReshapeOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace ReshapeOpInterface

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

namespace TypecastOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 mlir::tt::DataTypeAttr dtype,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             mlir::tt::DataTypeAttr dtype, llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace TypecastOpInterface

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace TransposeOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0,
                 const int dim1, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0,
             const int dim1, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace TransposeOpInterface

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

namespace MatmulOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                 llvm::ArrayRef<int64_t> inputShapeB,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
                 bool transposeB);

llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                    llvm::ArrayRef<int64_t> inputShapeB,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                    llvm::ArrayRef<int64_t> outputShape,
                                    mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                                    bool transposeA, bool transposeB);
}; // namespace MatmulOpInterface

//===----------------------------------------------------------------------===//
// MultiplyOp
//===----------------------------------------------------------------------===//

namespace MultiplyOpInterface {
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                 llvm::ArrayRef<int64_t> inputShapeB,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
             llvm::ArrayRef<int64_t> inputShapeB,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace MultiplyOpInterface

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
