// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace mlir::tt::ttnn::op_model {

// Checks if the tensor layout is legal for the given tensor shape.
bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 TTNNLayoutAttr layout,
                                 ttcore::GridAttr maxGrid);

// Calculate the output tensor type of the prepared weights for a conv2d op.
mlir::RankedTensorType getPreparedConv2dWeightsOutputTensor(Conv2dOp *op);

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

namespace Device {
llvm::Expected<bool> getDeviceConstraints(ttcore::GridAttr workerGrid);
}; // namespace Device

template <typename OpTy>
struct OpModel;

//===----------------------------------------------------------------------===//
// Unary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct UnaryEltwiseOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<ReluOp> : UnaryEltwiseOpModel<ReluOp> {};

template <>
struct OpModel<SqrtOp> : UnaryEltwiseOpModel<SqrtOp> {};

template <>
struct OpModel<SinOp> : UnaryEltwiseOpModel<SinOp> {};

template <>
struct OpModel<AbsOp> : UnaryEltwiseOpModel<AbsOp> {};

template <>
struct OpModel<CosOp> : UnaryEltwiseOpModel<CosOp> {};

template <>
struct OpModel<TanhOp> : UnaryEltwiseOpModel<TanhOp> {};

template <>
struct OpModel<LogOp> : UnaryEltwiseOpModel<LogOp> {};

template <>
struct OpModel<CeilOp> : UnaryEltwiseOpModel<CeilOp> {};

template <>
struct OpModel<SignOp> : UnaryEltwiseOpModel<SignOp> {};

template <>
struct OpModel<FloorOp> : UnaryEltwiseOpModel<FloorOp> {};

template <>
struct OpModel<IsFiniteOp> : UnaryEltwiseOpModel<IsFiniteOp> {};

template <>
struct OpModel<LogicalNotOp> : UnaryEltwiseOpModel<LogicalNotOp> {};

template <>
struct OpModel<NegOp> : UnaryEltwiseOpModel<NegOp> {};

template <>
struct OpModel<TanOp> : UnaryEltwiseOpModel<TanOp> {};

template <>
struct OpModel<AtanOp> : UnaryEltwiseOpModel<AtanOp> {};

template <>
struct OpModel<Log1pOp> : UnaryEltwiseOpModel<Log1pOp> {};

template <>
struct OpModel<Expm1Op> : UnaryEltwiseOpModel<Expm1Op> {};

template <>
struct OpModel<ReciprocalOp> : UnaryEltwiseOpModel<ReciprocalOp> {};

template <typename OpT>
struct UnaryEltwiseWithFastApproxModeOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<mlir::tt::ttnn::RsqrtOp>
    : UnaryEltwiseWithFastApproxModeOpModel<mlir::tt::ttnn::RsqrtOp> {};

template <>
struct OpModel<mlir::tt::ttnn::GeluOp>
    : UnaryEltwiseWithFastApproxModeOpModel<mlir::tt::ttnn::GeluOp> {};

template <>
struct OpModel<mlir::tt::ttnn::ExpOp>
    : UnaryEltwiseWithFastApproxModeOpModel<mlir::tt::ttnn::ExpOp> {};

template <>
struct OpModel<mlir::tt::ttnn::ErfOp>
    : UnaryEltwiseWithFastApproxModeOpModel<mlir::tt::ttnn::ErfOp> {};

template <>
struct OpModel<mlir::tt::ttnn::ErfcOp>
    : UnaryEltwiseWithFastApproxModeOpModel<mlir::tt::ttnn::ErfcOp> {};

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
// LeakyReluOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::LeakyReluOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat slope,
      mlir::tt::ttnn::TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
               mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat slope,
               mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// Binary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct BinaryEltwiseOpModel {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
      TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
      TTNNLayoutAttr inputLayoutB, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
               TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<AddOp> : BinaryEltwiseOpModel<AddOp> {};

template <>
struct OpModel<MultiplyOp> : BinaryEltwiseOpModel<MultiplyOp> {};

template <>
struct OpModel<SubtractOp> : BinaryEltwiseOpModel<SubtractOp> {};

template <>
struct OpModel<MaximumOp> : BinaryEltwiseOpModel<MaximumOp> {};

template <>
struct OpModel<MinimumOp> : BinaryEltwiseOpModel<MinimumOp> {};

template <>
struct OpModel<DivideOp> : BinaryEltwiseOpModel<DivideOp> {};

template <>
struct OpModel<EqualOp> : BinaryEltwiseOpModel<EqualOp> {};

template <>
struct OpModel<NotEqualOp> : BinaryEltwiseOpModel<NotEqualOp> {};

template <>
struct OpModel<GreaterEqualOp> : BinaryEltwiseOpModel<GreaterEqualOp> {};

template <>
struct OpModel<GreaterThanOp> : BinaryEltwiseOpModel<GreaterThanOp> {};

template <>
struct OpModel<LessEqualOp> : BinaryEltwiseOpModel<LessEqualOp> {};

template <>
struct OpModel<LessThanOp> : BinaryEltwiseOpModel<LessThanOp> {};

template <>
struct OpModel<LogicalAndOp> : BinaryEltwiseOpModel<LogicalAndOp> {};

template <>
struct OpModel<LogicalOrOp> : BinaryEltwiseOpModel<LogicalOrOp> {};

template <>
struct OpModel<LogicalXorOp> : BinaryEltwiseOpModel<LogicalXorOp> {};

//===----------------------------------------------------------------------===//
// Ternary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct TernaryEltwiseOpModel {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
      TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
      TTNNLayoutAttr inputLayoutB, llvm::ArrayRef<int64_t> inputShapeC,
      TTNNLayoutAttr inputLayoutC, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
               llvm::ArrayRef<int64_t> inputShapeC, TTNNLayoutAttr inputLayoutC,
               TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<WhereOp> : TernaryEltwiseOpModel<WhereOp> {};

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct ReductionOpModel {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, std::optional<llvm::ArrayRef<int64_t>> dimArg,
      bool keepDim, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
               TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<MeanOp> : ReductionOpModel<MeanOp> {};

template <>
struct OpModel<SumOp> : ReductionOpModel<SumOp> {};

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SoftmaxOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, const int dimArg,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             const int dimArg,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ReshapeOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> outputShape,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> outputShape,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SliceOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> begins,
                   llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
               llvm::ArrayRef<int64_t> step, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<TypecastOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, ttcore::DataTypeAttr dtype,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             ttcore::DataTypeAttr dtype,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ToLayoutOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, std::optional<ttcore::DataType> outputDtype,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<ttcore::DataType> outputDtype,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ToMemoryConfigOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, MemoryConfigAttr memoryConfig,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             MemoryConfigAttr memoryConfig,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ConcatOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   std::vector<llvm::ArrayRef<int64_t>> inputShapes,
                   std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(std::vector<llvm::ArrayRef<int64_t>> inputShapes,
               std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<TransposeOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             const int dim0, const int dim1,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<LinearOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
      TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
      TTNNLayoutAttr inputLayoutB,
      std::optional<llvm::ArrayRef<int64_t>> biasShape,
      std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
      bool transposeA, bool transposeB);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
               std::optional<llvm::ArrayRef<int64_t>> biasShape,
               std::optional<TTNNLayoutAttr> biasLayout,
               TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB);
};

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<MatmulOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
      TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
      TTNNLayoutAttr inputLayoutB, TTNNLayoutAttr outputLayout, bool transposeA,
      bool transposeB);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
               TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB);
};

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<Conv2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> weightShape,
      TTNNLayoutAttr weightLayout,
      std::optional<llvm::ArrayRef<int64_t>> biasShape,
      std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
      uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
      uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
      llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
      llvm::ArrayRef<int32_t> dilation, uint32_t groups,
      std::optional<Conv2dConfigAttr> conv2dConfig,
      std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(
      llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
      llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
      std::optional<llvm::ArrayRef<int64_t>> biasShape,
      std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
      uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
      uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
      llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
      llvm::ArrayRef<int32_t> dilation, uint32_t groups,
      std::optional<Conv2dConfigAttr> conv2dConfig,
      std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
      TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ConvTranspose2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> weightShape,
      TTNNLayoutAttr weightLayout,
      std::optional<llvm::ArrayRef<int64_t>> biasShape,
      std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
      uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
      uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
      llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
      llvm::ArrayRef<int32_t> output_padding, llvm::ArrayRef<int32_t> dilation,
      uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(
      llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
      llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
      std::optional<llvm::ArrayRef<int64_t>> biasShape,
      std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
      uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
      uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
      llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
      llvm::ArrayRef<int32_t> output_padding, llvm::ArrayRef<int32_t> dilation,
      uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
      TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<MaxPool2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, int32_t batchSize, int32_t inputHeight,
      int32_t inputWidth, int32_t inputChannels,
      llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
      llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
      bool ceilMode, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
               int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
               llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
               llvm::ArrayRef<int32_t> dilation, bool ceilMode,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ClampScalarOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ClampScalarOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, llvm::APFloat min,
                   llvm::APFloat max, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             llvm::APFloat min,
                                             llvm::APFloat max,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<PermuteOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> permutation,
      llvm::APFloat padValue, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<UpsampleOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, mlir::Attribute scaleFactor,
                   llvm::StringRef mode, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             mlir::Attribute scaleFactor,
                                             llvm::StringRef mode,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<EmbeddingOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> weightShape,
      TTNNLayoutAttr weightLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
               TTNNLayoutAttr outputLayout);
};

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
