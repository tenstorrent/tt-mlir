// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpModel/TTNN/NativeFromMLIR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#ifdef TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn::op_model {

namespace detail {
std::optional<::tt::target::ttnn::MemoryConfigT>
getNullableMemoryConfigT(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getMemoryConfigT(layout);
}

std::unique_ptr<::tt::target::ttnn::TensorRefT>
getOutputTensorRefT(TTNNLayoutAttr layout) {
  auto memoryConfigNative = getNullableMemoryConfigT(layout);
  if (!memoryConfigNative.has_value()) {
    return nullptr;
  }

  auto tensorRefNative = std::make_unique<::tt::target::ttnn::TensorRefT>();
  tensorRefNative->desc = std::make_unique<::tt::target::ttnn::TensorDescT>();
  tensorRefNative->desc->layout =
      std::make_unique<::tt::target::ttnn::LayoutDescT>();
  tensorRefNative->desc->layout->memory_desc =
      std::make_unique<::tt::target::ttnn::MemoryDescT>();
  tensorRefNative->desc->layout->memory_desc->memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          memoryConfigNative.value());

  tensorRefNative->desc->layout->memory_desc->data_type =
      toNative(layout.getDataType());

  return tensorRefNative;
}

/**
 * @brief Reorder pool2d padding from IR convention to tt-metal convention.
 *
 * IR stores padding as [H_low, W_low, H_high, W_high] (top, left, bottom,
 * right) but tt-metal expects [top, bottom, left, right] (H_low, H_high,
 * W_low, W_high). The runtime does this reordering when executing from
 * flatbuffers, but the op_model constraint query path must do it too.
 */
std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>
reorderPool2dPadding(llvm::ArrayRef<int32_t> padding) {
  if (padding.size() == 2) {
    return conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding);
  }
  return std::array<uint32_t, 4>{
      static_cast<uint32_t>(padding[0]), // top
      static_cast<uint32_t>(padding[2]), // bottom
      static_cast<uint32_t>(padding[1]), // left
      static_cast<uint32_t>(padding[3]), // right
  };
}
} // namespace detail

template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryOpT
buildEltwiseUnaryOpTFromMLIR(TTNNLayoutAttr outputLayout,
                             std::optional<llvm::APFloat> slope) {
  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOp;

  if (std::is_same_v<OpTy, TanhOp>) {
    eltwiseUnaryOp.type = ::tt::target::ttnn::EltwiseUnaryOpType::Tanh;
  } else if (std::is_same_v<OpTy, SigmoidOp>) {
    eltwiseUnaryOp.type = ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid;
  } else if (std::is_same_v<OpTy, LeakyReluOp>) {
    eltwiseUnaryOp.type = ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu;
    assert(slope.has_value() && "LeakyReluOp requires a slope value");
    ::tt::target::ttnn::EltwiseOpWithFloatParamsT
        eltwiseOpWithFloatParamsNative;
    eltwiseOpWithFloatParamsNative.parameter = slope.value().convertToFloat();
    eltwiseUnaryOp.params.Set(eltwiseOpWithFloatParamsNative);
  }

  eltwiseUnaryOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryOp;
}

template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOp;

  eltwiseUnaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryCompositeOp;
}

#define INSTANTIATE_BUILD_ELTWISE_UNARY(OP)                                    \
  template ::tt::target::ttnn::EltwiseUnaryOpT                                 \
      buildEltwiseUnaryOpTFromMLIR<OP>(TTNNLayoutAttr,                         \
                                       std::optional<llvm::APFloat>);

#define INSTANTIATE_BUILD_ELTWISE_UNARY_COMPOSITE(OP)                          \
  template ::tt::target::ttnn::EltwiseUnaryCompositeOpT                        \
      buildEltwiseUnaryCompositeOpTFromMLIR<OP>(TTNNLayoutAttr);

INSTANTIATE_BUILD_ELTWISE_UNARY(ReluOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(Relu6Op);
INSTANTIATE_BUILD_ELTWISE_UNARY(HardsigmoidOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SqrtOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SinOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AbsOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(CosOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(LogOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(CeilOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SignOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(FloorOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(IsFiniteOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(LogicalNotOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(NegOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(TanOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AtanOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AsinOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AsinhOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AcosOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ReciprocalOp);
INSTANTIATE_BUILD_ELTWISE_UNARY_COMPOSITE(CbrtOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(BitwiseNotOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SiluOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(MishOp);
INSTANTIATE_BUILD_ELTWISE_UNARY_COMPOSITE(Log1pOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(Expm1Op);
INSTANTIATE_BUILD_ELTWISE_UNARY(RsqrtOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ErfOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ErfcOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ExpOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(GeluOp);

INSTANTIATE_BUILD_ELTWISE_UNARY(TanhOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SigmoidOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(LeakyReluOp);

template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryOpT
buildEltwiseBinaryOpTFromMLIR(TTNNLayoutAttr outputLayout,
                              ttcore::DataTypeAttr opDtypeAttr) {
  ::tt::target::ttnn::EltwiseBinaryOpT eltwiseBinaryOp;

  eltwiseBinaryOp.out = detail::getOutputTensorRefT(outputLayout);
  if (eltwiseBinaryOp.out) {
    eltwiseBinaryOp.output_dtype =
        eltwiseBinaryOp.out->desc->layout->memory_desc->data_type;
  }
  if (opDtypeAttr) {
    eltwiseBinaryOp.output_dtype = toNative(opDtypeAttr.getValue());
    if (eltwiseBinaryOp.out && eltwiseBinaryOp.output_dtype.has_value()) {
      eltwiseBinaryOp.out->desc->layout->memory_desc->data_type =
          eltwiseBinaryOp.output_dtype.value();
    }
  }

  return eltwiseBinaryOp;
}

template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryCompositeOpT
buildEltwiseBinaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseBinaryCompositeOpT eltwiseBinaryCompositeOp;

  eltwiseBinaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseBinaryCompositeOp;
}

#define INSTANTIATE_BUILD_ELTWISE_BINARY(OP)                                   \
  template ::tt::target::ttnn::EltwiseBinaryOpT                                \
      buildEltwiseBinaryOpTFromMLIR<OP>(TTNNLayoutAttr, ttcore::DataTypeAttr);

#define INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(OP)                         \
  template ::tt::target::ttnn::EltwiseBinaryCompositeOpT                       \
      buildEltwiseBinaryCompositeOpTFromMLIR<OP>(TTNNLayoutAttr);

INSTANTIATE_BUILD_ELTWISE_BINARY(AddOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(MultiplyOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalRightShiftOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(SubtractOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(MaximumOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(MinimumOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(DivideOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(EqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(NotEqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(GreaterEqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(GreaterThanOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LessEqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LessThanOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalAndOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalOrOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalXorOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(PowTensorOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(RemainderOp);

INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(BitwiseAndOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(BitwiseOrOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(BitwiseXorOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(LogicalLeftShiftOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(Atan2Op);

::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
buildEltwiseBinaryCompositeScalarOpTFromMLIR(mlir::Attribute exponent,
                                             TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
      eltwiseBinaryCompositeScalarOp;
  eltwiseBinaryCompositeScalarOp.type =
      ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpType::PowScalar;

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(exponent)) {
    ::tt::target::ttnn::FloatingPointTypeT fp;
    fp.value = floatAttr.getValue().convertToFloat();
    eltwiseBinaryCompositeScalarOp.rhs.Set(fp);
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(exponent)) {
    ::tt::target::ttnn::IntegralTypeT i32;
    i32.value = static_cast<int32_t>(intAttr.getInt());
    eltwiseBinaryCompositeScalarOp.rhs.Set(i32);
  } else {
    llvm::report_fatal_error("Invalid exponent");
  }

  eltwiseBinaryCompositeScalarOp.out =
      detail::getOutputTensorRefT(outputLayout);

  return eltwiseBinaryCompositeScalarOp;
}

template <typename OpTy>
::tt::target::ttnn::EltwiseTernaryWhereOpT
buildEltwiseTernaryOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseTernaryWhereOpT eltwiseTernaryWhereOp;

  eltwiseTernaryWhereOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseTernaryWhereOp;
}

#define INSTANTIATE_BUILD_ELTWISE_TERNARY(OP)                                  \
  template ::tt::target::ttnn::EltwiseTernaryWhereOpT                          \
      buildEltwiseTernaryOpTFromMLIR<OP>(TTNNLayoutAttr);

INSTANTIATE_BUILD_ELTWISE_TERNARY(WhereOp);

template <typename OpTy>
::tt::target::ttnn::EltwiseQuantizationOpT
buildEltwiseQuantizationOpTFromMLIR(std::optional<int32_t> axis,
                                    std::optional<ttcore::DataType> outputDtype,
                                    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseQuantizationOpT eltwiseQuantizationOp;

  if constexpr (std::is_same_v<OpTy, QuantizeOp>) {
    eltwiseQuantizationOp.type =
        ::tt::target::ttnn::EltwiseQuantizationOpType::Quantize;
  } else if constexpr (std::is_same_v<OpTy, DequantizeOp>) {
    eltwiseQuantizationOp.type =
        ::tt::target::ttnn::EltwiseQuantizationOpType::Dequantize;
  } else if constexpr (std::is_same_v<OpTy, RequantizeOp>) {
    eltwiseQuantizationOp.type =
        ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize;
  } else {
    static_assert(ttmlir::utils::always_false(), "Unsupported OpTy");
  }

  eltwiseQuantizationOp.axis =
      axis.has_value() ? ::flatbuffers::Optional<int32_t>(axis.value())
                       : ::flatbuffers::nullopt;

  eltwiseQuantizationOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseQuantizationOp;
}

#define INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(OP)                             \
  template ::tt::target::ttnn::EltwiseQuantizationOpT                          \
      buildEltwiseQuantizationOpTFromMLIR<OP>(std::optional<int32_t>,          \
                                              std::optional<ttcore::DataType>, \
                                              TTNNLayoutAttr);

INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(QuantizeOp);
INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(DequantizeOp);
INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(RequantizeOp);

::tt::target::ttnn::LinearOpT buildLinearOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {

  ::tt::target::ttnn::LinearOpT linearOp;

  linearOp.transpose_a = transposeA;
  linearOp.transpose_b = transposeB;

  if (activation) {
    linearOp.activation = activation->str();
  }

  if (programConfigAttr.has_value()) {
    mlir::TypeSwitch<mlir::Attribute>(*programConfigAttr)
        .Case<MatmulMultiCoreReuseProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastProgramConfigAttr,
              MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            [&](auto config) {
              linearOp.matmul_program_config.Set(toNative(config));
            });
  }
  linearOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;

  linearOp.out = detail::getOutputTensorRefT(outputLayout);

  return linearOp;
}

::tt::target::ttnn::MatmulOpT buildMatmulOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {

  ::tt::target::ttnn::MatmulOpT matmulOp;

  matmulOp.transpose_a = transposeA;
  matmulOp.transpose_b = transposeB;

  if (activation) {
    matmulOp.activation = activation->str();
  }

  if (programConfigAttr.has_value()) {
    mlir::TypeSwitch<mlir::Attribute>(*programConfigAttr)
        .Case<MatmulMultiCoreReuseProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastProgramConfigAttr,
              MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            [&](auto config) {
              matmulOp.matmul_program_config.Set(toNative(config));
            });
  }
  matmulOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;

  matmulOp.out = detail::getOutputTensorRefT(outputLayout);

  return matmulOp;
}

::tt::target::ttnn::Conv2dOpT buildConv2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::Conv2dOpT conv2dOp;
  conv2dOp.in_channels = in_channels;
  conv2dOp.out_channels = out_channels;
  conv2dOp.batch_size = batch_size;
  conv2dOp.input_height = input_height;
  conv2dOp.input_width = input_width;
  conv2dOp.kernel_size =
      std::vector<int32_t>(kernel_size.begin(), kernel_size.end());
  conv2dOp.stride = std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&conv2dOp](const auto &arr) {
        conv2dOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  conv2dOp.dilation = std::vector<int32_t>(dilation.begin(), dilation.end());
  conv2dOp.groups = groups;
  conv2dOp.out = detail::getOutputTensorRefT(outputLayout);
  conv2dOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  conv2dOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  conv2dOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;

  return conv2dOp;
}

::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(mlir::Attribute min,
                                                 mlir::Attribute max,
                                                 TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOp;
  eltwiseUnaryCompositeOp.type =
      ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar;

  ::tt::target::ttnn::ClampScalarOpParamsT params;

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(min)) {
    ::tt::target::ttnn::FloatingPointTypeT fp;
    fp.value = floatAttr.getValue().convertToFloat();
    params.min.Set(fp);
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(min)) {
    ::tt::target::ttnn::IntegralTypeT i32;
    i32.value = static_cast<int32_t>(intAttr.getInt());
    params.min.Set(i32);
  } else {
    llvm_unreachable("Invalid clamp min attribute");
  }

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(max)) {
    ::tt::target::ttnn::FloatingPointTypeT fp;
    fp.value = floatAttr.getValue().convertToFloat();
    params.max.Set(fp);
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(max)) {
    ::tt::target::ttnn::IntegralTypeT i32;
    i32.value = static_cast<int32_t>(intAttr.getInt());
    params.max.Set(i32);
  } else {
    llvm_unreachable("Invalid clamp max attribute");
  }

  eltwiseUnaryCompositeOp.params.Set(params);

  eltwiseUnaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryCompositeOp;
}

::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOp;
  eltwiseUnaryCompositeOp.type =
      ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor;

  eltwiseUnaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryCompositeOp;
}

::tt::target::ttnn::Conv3dOpT buildConv3dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_depth, uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::StringRef padding_mode,
    uint32_t groups, std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::Conv3dOpT conv3dOp;
  conv3dOp.in_channels = in_channels;
  conv3dOp.out_channels = out_channels;
  conv3dOp.batch_size = batch_size;
  conv3dOp.input_depth = input_depth;
  conv3dOp.input_height = input_height;
  conv3dOp.input_width = input_width;
  conv3dOp.kernel_size =
      std::vector<int32_t>(kernel_size.begin(), kernel_size.end());
  conv3dOp.stride = std::vector<int32_t>(stride.begin(), stride.end());
  conv3dOp.padding = std::vector<int32_t>(padding.begin(), padding.end());
  conv3dOp.padding_mode = padding_mode.str();
  conv3dOp.groups = groups;
  conv3dOp.out = detail::getOutputTensorRefT(outputLayout);
  if (outputDtype.has_value() && outputDtype.value()) {
    conv3dOp.output_dtype = toNative(outputDtype.value().getValue());
  }
  conv3dOp.conv3d_config =
      (conv3dConfig.has_value() && *conv3dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv3dConfigT>(
                toNative(*conv3dConfig))
          : nullptr;
  conv3dOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  auto outputMemoryConfigT = detail::getNullableMemoryConfigT(outputLayout);
  conv3dOp.memory_config =
      outputMemoryConfigT.has_value()
          ? std::make_unique<::tt::target::ttnn::MemoryConfigT>(
                *outputMemoryConfigT)
          : nullptr;

  return conv3dOp;
}

::tt::target::ttnn::ConvTranspose2dOpT buildConvTranspose2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ConvTranspose2dOpT convTranspose2dOp;
  convTranspose2dOp.in_channels = in_channels;
  convTranspose2dOp.out_channels = out_channels;
  convTranspose2dOp.batch_size = batch_size;
  convTranspose2dOp.input_height = input_height;
  convTranspose2dOp.input_width = input_width;
  convTranspose2dOp.kernel_size =
      std::vector<int32_t>(kernel_size.begin(), kernel_size.end());
  convTranspose2dOp.stride = std::vector<int32_t>(stride.begin(), stride.end());
  convTranspose2dOp.padding =
      std::vector<int32_t>(padding.begin(), padding.end());
  convTranspose2dOp.output_padding =
      std::vector<int32_t>(output_padding.begin(), output_padding.end());
  convTranspose2dOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  convTranspose2dOp.groups = groups;
  convTranspose2dOp.out = detail::getOutputTensorRefT(outputLayout);
  convTranspose2dOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  convTranspose2dOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  convTranspose2dOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;

  return convTranspose2dOp;
}

::tt::target::ttnn::PrepareConv2dWeightsOpT
buildPrepareConv2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConv2dWeightsOpT prepareConv2dWeightsOp;
  prepareConv2dWeightsOp.in_channels = inChannels;
  prepareConv2dWeightsOp.out_channels = outChannels;
  prepareConv2dWeightsOp.batch_size = batchSize;
  prepareConv2dWeightsOp.input_height = inputHeight;
  prepareConv2dWeightsOp.input_width = inputWidth;
  prepareConv2dWeightsOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConv2dWeightsOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConv2dWeightsOp](const auto &arr) {
        prepareConv2dWeightsOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConv2dWeightsOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConv2dWeightsOp.has_bias = hasBias;
  prepareConv2dWeightsOp.groups = groups;
  prepareConv2dWeightsOp.weights_format = weightsFormat.str();
  prepareConv2dWeightsOp.input_tensor_layout = toNative(inputTensorLayout);
  prepareConv2dWeightsOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConv2dWeightsOp.output_dtype = toNative(outputDtype.value());
  }
  prepareConv2dWeightsOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConv2dWeightsOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConv2dWeightsOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConv2dWeightsOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConv2dWeightsOp.out = detail::getOutputTensorRefT(outputLayout);

  return prepareConv2dWeightsOp;
}

::tt::target::ttnn::PrepareConv2dBiasOpT buildPrepareConv2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConv2dBiasOpT prepareConv2dBiasOp;
  prepareConv2dBiasOp.in_channels = inChannels;
  prepareConv2dBiasOp.out_channels = outChannels;
  prepareConv2dBiasOp.batch_size = batchSize;
  prepareConv2dBiasOp.input_height = inputHeight;
  prepareConv2dBiasOp.input_width = inputWidth;
  prepareConv2dBiasOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConv2dBiasOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConv2dBiasOp](const auto &arr) {
        prepareConv2dBiasOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConv2dBiasOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConv2dBiasOp.groups = groups;
  prepareConv2dBiasOp.input_tensor_layout = toNative(inputTensorLayout);
  prepareConv2dBiasOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConv2dBiasOp.output_dtype = toNative(outputDtype.value());
  }
  prepareConv2dBiasOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConv2dBiasOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConv2dBiasOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConv2dBiasOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConv2dBiasOp.out = detail::getOutputTensorRefT(outputLayout);

  return prepareConv2dBiasOp;
}

::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
buildPrepareConvTranspose2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> outputPadding,
    llvm::ArrayRef<int32_t> dilation, bool hasBias, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
      prepareConvTranspose2dWeightsOp;
  prepareConvTranspose2dWeightsOp.in_channels = inChannels;
  prepareConvTranspose2dWeightsOp.out_channels = outChannels;
  prepareConvTranspose2dWeightsOp.batch_size = batchSize;
  prepareConvTranspose2dWeightsOp.input_height = inputHeight;
  prepareConvTranspose2dWeightsOp.input_width = inputWidth;
  prepareConvTranspose2dWeightsOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConvTranspose2dWeightsOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConvTranspose2dWeightsOp](const auto &arr) {
        prepareConvTranspose2dWeightsOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConvTranspose2dWeightsOp.output_padding =
      std::vector<int32_t>(outputPadding.begin(), outputPadding.end());
  prepareConvTranspose2dWeightsOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConvTranspose2dWeightsOp.has_bias = hasBias;
  prepareConvTranspose2dWeightsOp.groups = groups;
  prepareConvTranspose2dWeightsOp.weights_format = weightsFormat.str();
  prepareConvTranspose2dWeightsOp.mirror_kernel = mirrorKernel;
  prepareConvTranspose2dWeightsOp.input_tensor_layout =
      toNative(inputTensorLayout);
  prepareConvTranspose2dWeightsOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConvTranspose2dWeightsOp.output_dtype =
        toNative(outputDtype.value());
  }
  prepareConvTranspose2dWeightsOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConvTranspose2dWeightsOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConvTranspose2dWeightsOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConvTranspose2dWeightsOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConvTranspose2dWeightsOp.out =
      detail::getOutputTensorRefT(outputLayout);

  return prepareConvTranspose2dWeightsOp;
}

::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
buildPrepareConvTranspose2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
      prepareConvTranspose2dBiasOp;
  prepareConvTranspose2dBiasOp.in_channels = inChannels;
  prepareConvTranspose2dBiasOp.out_channels = outChannels;
  prepareConvTranspose2dBiasOp.batch_size = batchSize;
  prepareConvTranspose2dBiasOp.input_height = inputHeight;
  prepareConvTranspose2dBiasOp.input_width = inputWidth;
  prepareConvTranspose2dBiasOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConvTranspose2dBiasOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConvTranspose2dBiasOp](const auto &arr) {
        prepareConvTranspose2dBiasOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConvTranspose2dBiasOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConvTranspose2dBiasOp.groups = groups;
  prepareConvTranspose2dBiasOp.input_tensor_layout =
      toNative(inputTensorLayout);
  prepareConvTranspose2dBiasOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConvTranspose2dBiasOp.output_dtype = toNative(outputDtype.value());
  }
  prepareConvTranspose2dBiasOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConvTranspose2dBiasOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConvTranspose2dBiasOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConvTranspose2dBiasOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConvTranspose2dBiasOp.out = detail::getOutputTensorRefT(outputLayout);

  return prepareConvTranspose2dBiasOp;
}

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_ENABLE_OPMODEL
