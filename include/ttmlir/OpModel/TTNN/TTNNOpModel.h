// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <tuple>

namespace mlir::tt::op_model::ttnn {

// Checks if the tensor layout is legal for the given tensor shape.
bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr layout,
                                 GridAttr maxGrid);

// Calculate the output tensor type of the prepared weights for a conv2d op.
mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(mlir::tt::ttnn::Conv2dOp *op);

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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
// SqrtOp
//===----------------------------------------------------------------------===//

namespace SqrtOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace SqrtOpInterface

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//

namespace SigmoidOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace SigmoidOpInterface

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

namespace AddOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
// ToLayoutOp
//===----------------------------------------------------------------------===//

namespace ToLayoutOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 std::optional<mlir::tt::DataType> outputDtype,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                 bool passDevicePtr);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             std::optional<mlir::tt::DataType> outputDtype,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool passDevicePtr);

}; // namespace ToLayoutOpInterface

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace TransposeOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
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

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

namespace Conv2dOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> weightShape,
                 mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
                 std::optional<llvm::ArrayRef<int64_t>> biasShape,
                 std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
                 uint32_t in_channels, uint32_t out_channels,
                 uint32_t batch_size, uint32_t input_height,
                 uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
                 llvm::ArrayRef<int32_t> stride,
                 llvm::ArrayRef<int32_t> padding,
                 llvm::ArrayRef<int32_t> dilation, uint32_t groups,
                 std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> weightShape,
             mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
             std::optional<llvm::ArrayRef<int64_t>> biasShape,
             std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
             uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
             uint32_t input_height, uint32_t input_width,
             llvm::ArrayRef<int32_t> kernel_size,
             llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
             llvm::ArrayRef<int32_t> dilation, uint32_t groups,
             std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace Conv2dOpInterface

//===----------------------------------------------------------------------===//
// MaxPool2D
//===----------------------------------------------------------------------===//
namespace MaxPool2DOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
                 int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
                 llvm::ArrayRef<int32_t> kernelSize,
                 llvm::ArrayRef<int32_t> stride,
                 llvm::ArrayRef<int32_t> padding,
                 llvm::ArrayRef<int32_t> dilation, bool ceilMode,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
             int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
             llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
             llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
             bool ceilMode, llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace MaxPool2DOpInterface

//===----------------------------------------------------------------------===//
// ClampScalar
//===----------------------------------------------------------------------===//
namespace ClampScalarOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
                 llvm::APFloat max, llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
             llvm::APFloat max, llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace ClampScalarOpInterface

//===----------------------------------------------------------------------===//
// Permute
//===----------------------------------------------------------------------===//
namespace PermuteOpInterface {
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace PermuteOpInterface

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
