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

template <typename OpTy>
struct OpModel;

//===----------------------------------------------------------------------===//
// Unary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct UnaryEltwiseOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<mlir::tt::ttnn::ReluOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::ReluOp> {};

template <>
struct OpModel<mlir::tt::ttnn::SqrtOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::SqrtOp> {};

template <>
struct OpModel<mlir::tt::ttnn::SinOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::SinOp> {};

template <>
struct OpModel<mlir::tt::ttnn::CosOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::CosOp> {};

template <>
struct OpModel<mlir::tt::ttnn::TanhOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::TanhOp> {};

template <>
struct OpModel<mlir::tt::ttnn::LogOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::LogOp> {};

template <>
struct OpModel<mlir::tt::ttnn::ReciprocalOp>
    : UnaryEltwiseOpModel<mlir::tt::ttnn::ReciprocalOp> {};

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::SigmoidOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ExpOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ExpOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// Binary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct BinaryEltwiseOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShapeA,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                   llvm::ArrayRef<int64_t> inputShapeB,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<mlir::tt::ttnn::AddOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::AddOp> {};

template <>
struct OpModel<mlir::tt::ttnn::MultiplyOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::MultiplyOp> {};

template <>
struct OpModel<mlir::tt::ttnn::SubtractOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::SubtractOp> {};

template <>
struct OpModel<mlir::tt::ttnn::MaximumOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::MaximumOp> {};

template <>
struct OpModel<mlir::tt::ttnn::MinimumOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::MinimumOp> {};

template <>
struct OpModel<mlir::tt::ttnn::DivideOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::DivideOp> {};

template <>
struct OpModel<mlir::tt::ttnn::EqualOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::EqualOp> {};

template <>
struct OpModel<mlir::tt::ttnn::NotEqualOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::NotEqualOp> {};

template <>
struct OpModel<mlir::tt::ttnn::GreaterEqualOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::GreaterEqualOp> {};

template <>
struct OpModel<mlir::tt::ttnn::GreaterThanOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::GreaterThanOp> {};

template <>
struct OpModel<mlir::tt::ttnn::LessEqualOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::LessEqualOp> {};

template <>
struct OpModel<mlir::tt::ttnn::LessThanOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::LessThanOp> {};

template <>
struct OpModel<mlir::tt::ttnn::LogicalAndOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::LogicalAndOp> {};

template <>
struct OpModel<mlir::tt::ttnn::LogicalOrOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::LogicalOrOp> {};

template <>
struct OpModel<mlir::tt::ttnn::LogicalXorOp>
    : BinaryEltwiseOpModel<mlir::tt::ttnn::LogicalXorOp> {};

//===----------------------------------------------------------------------===//
// Ternary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct TernaryEltwiseOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShapeA,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                   llvm::ArrayRef<int64_t> inputShapeB,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                   llvm::ArrayRef<int64_t> inputShapeC,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutC,
                   llvm::ArrayRef<int64_t> outputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
               llvm::ArrayRef<int64_t> inputShapeC,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutC,
               llvm::ArrayRef<int64_t> outputShape,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<mlir::tt::ttnn::WhereOp>
    : TernaryEltwiseOpModel<mlir::tt::ttnn::WhereOp> {};

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct ReductionOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<mlir::tt::ttnn::MeanOp>
    : ReductionOpModel<mlir::tt::ttnn::MeanOp> {};

template <>
struct OpModel<mlir::tt::ttnn::SumOp>
    : ReductionOpModel<mlir::tt::ttnn::SumOp> {};

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::SoftmaxOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ReshapeOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   llvm::ArrayRef<int64_t> outputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> outputShape,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::SliceOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
                   llvm::ArrayRef<int64_t> step,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
               llvm::ArrayRef<int64_t> step,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::TypecastOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   mlir::tt::ttcore::DataTypeAttr dtype,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               mlir::tt::ttcore::DataTypeAttr dtype,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ToLayoutOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   std::optional<mlir::tt::ttcore::DataType> outputDtype,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               std::optional<mlir::tt::ttcore::DataType> outputDtype,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ToMemoryConfigOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ConcatOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   std::vector<llvm::ArrayRef<int64_t>> inputShapes,
                   std::vector<mlir::tt::ttnn::TTNNLayoutAttr> inputLayouts,
                   const int dim, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(std::vector<llvm::ArrayRef<int64_t>> inputShapes,
               std::vector<mlir::tt::ttnn::TTNNLayoutAttr> inputLayouts,
               const int dim, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::TransposeOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0,
                   const int dim1, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0,
               const int dim1, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::LinearOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShapeA,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                   llvm::ArrayRef<int64_t> inputShapeB,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                   std::optional<llvm::ArrayRef<int64_t>> biasShape,
                   std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
                   bool transposeB);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
               std::optional<llvm::ArrayRef<int64_t>> biasShape,
               std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
               bool transposeB);
};

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::MatmulOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShapeA,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                   llvm::ArrayRef<int64_t> inputShapeB,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
                   bool transposeB);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
               bool transposeB);
};

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::Conv2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
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
      std::optional<mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
          deviceComputeKernelConfig,
      mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
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
               std::optional<mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
                   deviceComputeKernelConfig,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ConvTranspose2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
      mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
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
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::MaxPool2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
      int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
      llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
      llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
      bool ceilMode, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
               int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
               llvm::ArrayRef<int32_t> kernelSize,
               llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
               llvm::ArrayRef<int32_t> dilation, bool ceilMode,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ClampScalarOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::ClampScalarOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
      llvm::APFloat max, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
               llvm::APFloat max, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::PermuteOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::UpsampleOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::Attribute scaleFactor,
      llvm::StringRef mode, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               mlir::Attribute scaleFactor, llvm::StringRef mode,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::EmbeddingOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   llvm::ArrayRef<int64_t> weightShape,
                   mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> weightShape,
               mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<mlir::tt::ttnn::EmptyOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      mlir::tt::ttcore::DataTypeAttr dtype, mlir::tt::ttnn::Layout inputLayout,
      mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
      mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<mlir::tt::ttnn::ArangeOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   ::mlir::IntegerAttr start, ::mlir::IntegerAttr end,
                   ::mlir::IntegerAttr step,
                   std::optional<mlir::tt::ttcore::DataType> dtype,
                   std::optional<mlir::tt::ttnn::MemoryConfigAttr> memConfig,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ZerosOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<mlir::tt::ttnn::ZerosOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   mlir::tt::ttnn::ShapeAttr shape,
                   std::optional<mlir::tt::ttcore::DataType> dtype,
                   std::optional<mlir::tt::ttnn::Layout> layout,
                   std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};
} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
