// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpModelInterface.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace mlir::tt::op_model::ttnn {

// Checks if the tensor layout is legal for the given tensor shape.
bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr layout,
                                 mlir::tt::ttcore::GridAttr maxGrid);

// Calculate the output tensor type of the prepared weights for a conv2d op.
mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(mlir::tt::ttnn::Conv2dOp *op);

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

namespace Device {
llvm::Expected<bool>
getDeviceConstraints(mlir::tt::ttcore::GridAttr workerGrid);
}; // namespace Device

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

namespace ReluOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShapeA,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
// SliceOp
//===----------------------------------------------------------------------===//

namespace SliceOpInterface {
llvm::Expected<OpConstraints> getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> begins,
    llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
             llvm::ArrayRef<int64_t> step, llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace SliceOpInterface

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

namespace TypecastOpInterface {
llvm::Expected<OpConstraints> getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttcore::DataTypeAttr dtype, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             mlir::tt::ttcore::DataTypeAttr dtype,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace TypecastOpInterface

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

namespace ToLayoutOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 std::optional<mlir::tt::ttcore::DataType> outputDtype,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             std::optional<mlir::tt::ttcore::DataType> outputDtype,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace ToLayoutOpInterface

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

namespace ConcatOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 std::vector<llvm::ArrayRef<int64_t>> inputShapes,
                 std::vector<mlir::tt::ttnn::TTNNLayoutAttr> inputLayouts,
                 const int dim, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(std::vector<llvm::ArrayRef<int64_t>> inputShapes,
             std::vector<mlir::tt::ttnn::TTNNLayoutAttr> inputLayouts,
             const int dim, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace ConcatOpInterface

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

namespace TransposeOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShapeA,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShapeA,
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
llvm::Expected<OpConstraints> getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups,
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
// ConvTranspose2d
//===----------------------------------------------------------------------===//
namespace ConvTranspose2dOpInterface {
llvm::Expected<OpConstraints> getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
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
             llvm::ArrayRef<int32_t> output_padding,
             llvm::ArrayRef<int32_t> dilation, uint32_t groups,
             std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace ConvTranspose2dOpInterface

//===----------------------------------------------------------------------===//
// MaxPool2D
//===----------------------------------------------------------------------===//
namespace MaxPool2DOpInterface {
llvm::Expected<OpConstraints> getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, llvm::ArrayRef<int64_t> outputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
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

//===----------------------------------------------------------------------===//
// Upsample
//===----------------------------------------------------------------------===//
namespace UpsampleOpInterface {
llvm::Expected<OpConstraints> getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::Attribute scaleFactor,
    llvm::StringRef mode, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             mlir::Attribute scaleFactor, llvm::StringRef mode,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace UpsampleOpInterface

//===----------------------------------------------------------------------===//
// Subtract
//===----------------------------------------------------------------------===//
namespace SubtractOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShapeA,
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
} // namespace SubtractOpInterface

//===----------------------------------------------------------------------===//
// Maximum
//===----------------------------------------------------------------------===//
namespace MaximumOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShapeA,
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
} // namespace MaximumOpInterface

//===----------------------------------------------------------------------===//
// Minimum
//===----------------------------------------------------------------------===//
namespace MinimumOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShapeA,
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
} // namespace MinimumOpInterface

//===----------------------------------------------------------------------===//
// Embedding
//===----------------------------------------------------------------------===//
namespace EmbeddingOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> weightShape,
                 mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> weightShape,
             mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
} // namespace EmbeddingOpInterface

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//
namespace SumOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

}; // namespace SumOpInterface

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//
namespace SinOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace SinOpInterface

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//
namespace CosOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace CosOpInterface

//===----------------------------------------------------------------------===//
// ReciprocalOp
//===----------------------------------------------------------------------===//
namespace ReciprocalOpInterface {
llvm::Expected<OpConstraints>
getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                 llvm::ArrayRef<int64_t> inputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

llvm::Expected<size_t>
getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
             llvm::ArrayRef<int64_t> outputShape,
             mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
}; // namespace ReciprocalOpInterface

} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
