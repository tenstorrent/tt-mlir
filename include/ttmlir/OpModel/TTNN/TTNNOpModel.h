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
// Conv2dConfigAttr is used to determine the output tensor type.
mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(Conv2dOp *op,
                                     Conv2dConfigAttr conv2dConfig);

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
struct OpModel<ReluOp> : UnaryEltwiseOpModel<ReluOp> {};

template <>
struct OpModel<Relu6Op> : UnaryEltwiseOpModel<Relu6Op> {};

template <>
struct OpModel<HardsigmoidOp> : UnaryEltwiseOpModel<HardsigmoidOp> {};

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
struct OpModel<LogOp> : UnaryEltwiseWithFastApproxModeOpModel<LogOp> {};

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
struct OpModel<Log1pOp> : UnaryEltwiseWithFastApproxModeOpModel<Log1pOp> {};

template <>
struct OpModel<Expm1Op> : UnaryEltwiseOpModel<Expm1Op> {};

template <>
struct OpModel<ReciprocalOp> : UnaryEltwiseOpModel<ReciprocalOp> {};

template <>
struct OpModel<CbrtOp> : UnaryEltwiseOpModel<CbrtOp> {};

template <>
struct OpModel<BitwiseNotOp> : UnaryEltwiseOpModel<BitwiseNotOp> {};

template <>
struct OpModel<SiluOp> : UnaryEltwiseOpModel<SiluOp> {};

template <>
struct OpModel<MishOp> : UnaryEltwiseOpModel<MishOp> {};

template <>
struct OpModel<RsqrtOp> : UnaryEltwiseOpModel<RsqrtOp> {};

template <>
struct OpModel<GeluOp> : UnaryEltwiseWithFastApproxModeOpModel<GeluOp> {};

template <>
struct OpModel<ExpOp> : UnaryEltwiseWithFastApproxModeOpModel<ExpOp> {};

template <>
struct OpModel<ErfOp> : UnaryEltwiseWithFastApproxModeOpModel<ErfOp> {};

template <>
struct OpModel<ErfcOp> : UnaryEltwiseWithFastApproxModeOpModel<ErfcOp> {};

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SigmoidOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<LeakyReluOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, llvm::APFloat slope,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             llvm::APFloat slope,
                                             TTNNLayoutAttr outputLayout);
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

template <typename OpT>
struct BinaryCompositeOpModel {
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
struct OpModel<LogicalRightShiftOp>
    : BinaryEltwiseOpModel<LogicalRightShiftOp> {};

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

template <>
struct OpModel<PowTensorOp> : BinaryEltwiseOpModel<PowTensorOp> {};

template <>
struct OpModel<BitwiseAndOp> : BinaryCompositeOpModel<BitwiseAndOp> {};

template <>
struct OpModel<LogicalLeftShiftOp>
    : BinaryCompositeOpModel<LogicalLeftShiftOp> {};

template <>
struct OpModel<BitwiseOrOp> : BinaryCompositeOpModel<BitwiseOrOp> {};

template <>
struct OpModel<BitwiseXorOp> : BinaryCompositeOpModel<BitwiseXorOp> {};

template <>
struct OpModel<RemainderOp> : BinaryCompositeOpModel<RemainderOp> {};

template <>
struct OpModel<Atan2Op> : BinaryCompositeOpModel<Atan2Op> {};

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

template <>
struct OpModel<MaxOp> : ReductionOpModel<MaxOp> {};

template <>
struct OpModel<MinOp> : ReductionOpModel<MinOp> {};

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ArgMaxOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, std::optional<int32_t> dim,
                   bool keepDim, bool multicore, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             std::optional<int32_t> dim,
                                             bool keepDim, bool multicore,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ProdOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ProdOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, std::optional<int64_t> dim,
                   bool keepDim, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// Named Full Ops
//===----------------------------------------------------------------------===//

template <typename OpT>
struct NamedFullOpModel {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   mlir::tt::ttnn::ShapeAttr shape,
                   std::optional<mlir::tt::ttcore::DataType> dtype,
                   std::optional<mlir::tt::ttnn::Layout> layout,
                   std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<ZerosOp> : NamedFullOpModel<ZerosOp> {};

template <>
struct OpModel<OnesOp> : NamedFullOpModel<OnesOp> {};

//===----------------------------------------------------------------------===//
// Quantization Ops
//===----------------------------------------------------------------------===//
template <typename OpT>
struct QuantizationOpModel {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> scaleShape,
      TTNNLayoutAttr scaleLayout, llvm::ArrayRef<int64_t> zeroPointShape,
      TTNNLayoutAttr zeroPointLayout, std::optional<int32_t> axis,
      std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> scaleShape, TTNNLayoutAttr scaleLayout,
               llvm::ArrayRef<int64_t> zeroPointShape,
               TTNNLayoutAttr zeroPointLayout, std::optional<int32_t> axis,
               std::optional<ttcore::DataType> outputDtype,
               TTNNLayoutAttr outputLayout);
};

template <>
struct OpModel<QuantizeOp> : QuantizationOpModel<QuantizeOp> {};

template <>
struct OpModel<DequantizeOp> : QuantizationOpModel<DequantizeOp> {};

template <>
struct OpModel<RequantizeOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> inScaleShape,
      TTNNLayoutAttr inScaleLayout, llvm::ArrayRef<int64_t> inZeroPointShape,
      TTNNLayoutAttr inZeroPointLayout, llvm::ArrayRef<int64_t> outScaleShape,
      TTNNLayoutAttr outScaleLayout, llvm::ArrayRef<int64_t> outZeroPointShape,
      TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
      std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(
      llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
      llvm::ArrayRef<int64_t> inScaleShape, TTNNLayoutAttr inScaleLayout,
      llvm::ArrayRef<int64_t> inZeroPointShape,
      TTNNLayoutAttr inZeroPointLayout, llvm::ArrayRef<int64_t> outScaleShape,
      TTNNLayoutAttr outScaleLayout, llvm::ArrayRef<int64_t> outZeroPointShape,
      TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
      std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SoftmaxOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, const int dimArg,
                   bool numericStable, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             const int dimArg,
                                             bool numericStable,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ScatterOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
      TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
      TTNNLayoutAttr inputLayoutB, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
               llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
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
// SliceStaticOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SliceStaticOp> {
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
// SliceDynamicOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SliceDynamicOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> beginsShape,
      TTNNLayoutAttr beginsLayout, llvm::ArrayRef<int64_t> endsShape,
      TTNNLayoutAttr endsLayout, std::optional<llvm::SmallVector<int64_t>> step,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> beginsShape, TTNNLayoutAttr beginsLayout,
               llvm::ArrayRef<int64_t> endsShape, TTNNLayoutAttr endsLayout,
               std::optional<llvm::SmallVector<int64_t>> step,
               TTNNLayoutAttr outputLayout);
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
// MorehCumSumOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<MorehCumSumOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, const int64_t dim,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             const int64_t dim,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ConcatenateHeadsOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<ScaledDotProductAttentionDecodeOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> queryShape,
      TTNNLayoutAttr queryLayout, llvm::ArrayRef<int64_t> keyShape,
      TTNNLayoutAttr keyLayout, llvm::ArrayRef<int64_t> valueShape,
      TTNNLayoutAttr valueLayout, bool isCausal,
      std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
      std::optional<TTNNLayoutAttr> attentionMaskLayout,
      llvm::ArrayRef<int64_t> curPosTensorShape,
      TTNNLayoutAttr curPosTensorLayout,
      std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
      std::optional<TTNNLayoutAttr> attentionSinkLayout,
      std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
               llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
               llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
               bool isCausal,
               std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
               std::optional<TTNNLayoutAttr> attentionMaskLayout,
               llvm::ArrayRef<int64_t> curPosTensorShape,
               TTNNLayoutAttr curPosTensorLayout,
               std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
               std::optional<TTNNLayoutAttr> attentionSinkLayout,
               std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<ScaledDotProductAttentionOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> queryShape,
      TTNNLayoutAttr queryLayout, llvm::ArrayRef<int64_t> keyShape,
      TTNNLayoutAttr keyLayout, llvm::ArrayRef<int64_t> valueShape,
      TTNNLayoutAttr valueLayout,
      std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
      std::optional<TTNNLayoutAttr> attentionMaskLayout, bool isCausal,
      std::optional<llvm::APFloat> scale,
      std::optional<uint32_t> slidingWindowSize, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
               llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
               llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
               std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
               std::optional<TTNNLayoutAttr> attentionMaskLayout, bool isCausal,
               std::optional<llvm::APFloat> scale,
               std::optional<uint32_t> slidingWindowSize,
               TTNNLayoutAttr outputLayout);
};

//===-----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOp
// ===----------------------------------------------------------------------===//

template <>
struct OpModel<RotaryEmbeddingLlamaOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> cosShape,
      TTNNLayoutAttr cosLayout, llvm::ArrayRef<int64_t> sinShape,
      TTNNLayoutAttr sinLayout, llvm::ArrayRef<int64_t> transMatShape,
      TTNNLayoutAttr transMatLayout, bool isDecodeMode,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
               llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
               llvm::ArrayRef<int64_t> transMatShape,
               TTNNLayoutAttr transMatLayout, bool isDecodeMode,
               TTNNLayoutAttr outputLayout);
};

//===-----------------------------------------------------------------------===//
// NLPCreateQKVHeadsDecodeOp
// ===----------------------------------------------------------------------===//

template <>
struct OpModel<NLPCreateQKVHeadsDecodeOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout,
      std::optional<llvm::ArrayRef<int64_t>> batchOffsetShape,
      std::optional<TTNNLayoutAttr> batchOffsetLayout, uint32_t numHeads,
      std::optional<uint32_t> numKVHeads, std::optional<bool> overlapQKCoregrid,
      std::optional<uint32_t> sliceSize, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> batchOffsetShape,
               std::optional<TTNNLayoutAttr> batchOffsetLayout,
               uint32_t numHeads, std::optional<uint32_t> numKVHeads,
               std::optional<bool> overlapQKCoregrid,
               std::optional<uint32_t> sliceSize, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<SplitQueryKeyValueAndSplitHeadsOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout,
                   std::optional<llvm::ArrayRef<int64_t>> inputKVShape,
                   std::optional<TTNNLayoutAttr> inputKVLayout,
                   uint32_t numHeads, std::optional<uint32_t> numKVHeads,
                   bool transposeKey, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> inputKVShape,
               std::optional<TTNNLayoutAttr> inputKVLayout, uint32_t numHeads,
               std::optional<uint32_t> numKVHeads, bool transposeKey,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// NLPConcatHeadsOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<NLPConcatHeadsOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOp
//===----------------------------------------------------------------------===//
template <>
struct OpModel<NLPConcatHeadsDecodeOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, uint32_t headDim,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             uint32_t headDim,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<RepeatInterleaveOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, const unsigned int repeats,
                   const int dim, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             const unsigned int repeats,
                                             const int dim,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<RepeatOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> repeats,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             llvm::ArrayRef<int64_t> repeats,
                                             TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<PadOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int32_t> padding,
      llvm::APFloat padValue, bool multicore, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int32_t> padding, llvm::APFloat padValue,
               bool multicore, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<SortOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, int dim, bool descending,
                   bool stable, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             int dim, bool descending,
                                             bool stable,
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
// DeallocateOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<DeallocateOp> {
  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             bool force);
};

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<FillCacheOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> cacheShape,
      TTNNLayoutAttr cacheLayout, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, uint32_t batchOffset,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
               llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               uint32_t batchOffset, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<UpdateCacheOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> cacheShape,
      TTNNLayoutAttr cacheLayout, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> updateIndexShape,
      TTNNLayoutAttr updateIndexLayout, uint32_t batchOffset,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
               llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> updateIndexShape,
               TTNNLayoutAttr updateIndexLayout, uint32_t batchOffset,
               TTNNLayoutAttr outputLayout);
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
// PrepareConv2dWeightsOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<PrepareConv2dWeightsOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, TTNNLayoutAttr weightLayout,
      llvm::ArrayRef<int64_t> weightShape, MemoryConfigAttr inputMemConfig,
      ::mlir::tt::ttnn::Layout inputTensorLayout, llvm::StringRef weightsFormat,
      int32_t inChannels, int32_t outChannels, int32_t batchSize,
      int32_t inputHeight, int32_t inputWidth,
      llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
      llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
      bool hasBias, int32_t groups, ttcore::DataType inputDtype,
      std::optional<ttcore::DataType> outputDtype,
      std::optional<Conv2dConfigAttr> conv2dConfig,
      std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
      TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<PrepareConv2dBiasOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, TTNNLayoutAttr biasLayout,
      llvm::ArrayRef<int64_t> biasShape, MemoryConfigAttr inputMemConfig,
      ::mlir::tt::ttnn::Layout inputTensorLayout, int32_t inChannels,
      int32_t outChannels, int32_t batchSize, int32_t inputHeight,
      int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
      llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
      llvm::ArrayRef<int32_t> dilation, int32_t groups,
      ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
      std::optional<Conv2dConfigAttr> conv2dConfig,
      std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
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
      bool ceilMode, bool inPlaceHalo, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
               int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
               llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
               llvm::ArrayRef<int32_t> dilation, bool ceilMode,
               bool inPlaceHalo, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// AvgPool2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<AvgPool2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, int32_t batchSize, int32_t inputHeight,
      int32_t inputWidth, int32_t inputChannels,
      llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
      llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
      bool ceilMode, bool inPlaceHalo, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
               int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
               llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
               llvm::ArrayRef<int32_t> dilation, bool ceilMode,
               bool inPlaceHalo, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// GlobalAvgPool2dOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<GlobalAvgPool2dOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, std::optional<ttcore::DataType> dtype,
      TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<ttcore::DataType> dtype,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<BatchNormInferenceOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout,
                   std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
                   std::optional<TTNNLayoutAttr> runningMeanLayout,
                   std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
                   std::optional<TTNNLayoutAttr> runningVarLayout,
                   std::optional<llvm::ArrayRef<int64_t>> weightShape,
                   std::optional<TTNNLayoutAttr> weightLayout,
                   std::optional<llvm::ArrayRef<int64_t>> biasShape,
                   std::optional<TTNNLayoutAttr> biasLayout,
                   llvm::APFloat epsilon, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
               std::optional<TTNNLayoutAttr> runningMeanLayout,
               std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
               std::optional<TTNNLayoutAttr> runningVarLayout,
               std::optional<llvm::ArrayRef<int64_t>> weightShape,
               std::optional<TTNNLayoutAttr> weightLayout,
               std::optional<llvm::ArrayRef<int64_t>> biasShape,
               std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
               TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<BatchNormTrainingOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout,
      std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
      std::optional<TTNNLayoutAttr> runningMeanLayout,
      std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
      std::optional<TTNNLayoutAttr> runningVarLayout,
      std::optional<llvm::ArrayRef<int64_t>> weightShape,
      std::optional<TTNNLayoutAttr> weightLayout,
      std::optional<llvm::ArrayRef<int64_t>> biasShape,
      std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
      llvm::APFloat momentum, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
               std::optional<TTNNLayoutAttr> runningMeanLayout,
               std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
               std::optional<TTNNLayoutAttr> runningVarLayout,
               std::optional<llvm::ArrayRef<int64_t>> weightShape,
               std::optional<TTNNLayoutAttr> weightLayout,
               std::optional<llvm::ArrayRef<int64_t>> biasShape,
               std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
               llvm::APFloat momentum, TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<RMSNormOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout,
                   std::optional<llvm::ArrayRef<int64_t>> weightShape,
                   std::optional<TTNNLayoutAttr> weightLayout,
                   std::optional<llvm::ArrayRef<int64_t>> biasShape,
                   std::optional<TTNNLayoutAttr> biasLayout,
                   llvm::APFloat epsilon, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               std::optional<llvm::ArrayRef<int64_t>> weightShape,
               std::optional<TTNNLayoutAttr> weightLayout,
               std::optional<llvm::ArrayRef<int64_t>> biasShape,
               std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
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
// ClampTensorOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ClampTensorOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> minShape,
                   TTNNLayoutAttr minLayout, llvm::ArrayRef<int64_t> maxShape,
                   TTNNLayoutAttr maxLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> minShape, TTNNLayoutAttr minLayout,
               llvm::ArrayRef<int64_t> maxShape, TTNNLayoutAttr maxLayout,
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
// PowScalarOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<PowScalarOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout, mlir::Attribute exponent,
                   TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t> getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                             TTNNLayoutAttr inputLayout,
                                             mlir::Attribute exponent,
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

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<EmbeddingBackwardOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
      TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> weightShape,
      TTNNLayoutAttr weightLayout, llvm::ArrayRef<int64_t> inGradientShape,
      TTNNLayoutAttr inGradientLayout, TTNNLayoutAttr outputLayout);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
               llvm::ArrayRef<int64_t> inGradientShape,
               TTNNLayoutAttr inGradientLayout, TTNNLayoutAttr outputLayout);
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
// FullOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::FullOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   mlir::tt::ttnn::ShapeAttr shape, mlir::Attribute fillValue,
                   std::optional<mlir::tt::ttcore::DataType> dtype,
                   std::optional<mlir::tt::ttnn::Layout> layout,
                   std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<ConstantOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(ttcore::GridAttr deviceGrid, mlir::ElementsAttr value,
                   TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// RandOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::RandOp> {
  static llvm::Expected<OpConstraints> getOpConstraints(
      mlir::tt::ttcore::GridAttr deviceGrid, mlir::tt::ttnn::ShapeAttr size,
      mlir::tt::ttcore::DataType dtype,
      mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
      mlir::tt::ttnn::Layout layout, llvm::APFloat low, llvm::APFloat high,
      uint32_t seed, mlir::tt::ttnn::TTNNLayoutAttr outputLayout);
};

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

template <>
struct OpModel<mlir::tt::ttnn::AssignOp> {
  static llvm::Expected<OpConstraints>
  getOpConstraints(mlir::tt::ttcore::GridAttr deviceGrid,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout,
                   mlir::tt::ttnn::MemoryConfigAttr outputMemConfig,
                   std::optional<mlir::tt::ttcore::DataType> outputDtype);

  static llvm::Expected<size_t>
  getOpRuntime(llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
               mlir::tt::ttnn::MemoryConfigAttr outputMemConfig,
               std::optional<mlir::tt::ttcore::DataType> outputDtype);
};

} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_OPMODEL_TTNN_TTNNOPMODEL_H
