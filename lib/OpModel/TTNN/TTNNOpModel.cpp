// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Utils.h"

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn::op_model {

#ifdef TTMLIR_ENABLE_OPMODEL
namespace operation {

/**
 * @brief Executes a constraint query and validates the response.
 *
 * This helper function attempts to execute the provided callable to obtain
 * constraint query information. It handles exceptions and validates that the
 * response contains the required data.
 *
 * @param callable A callable object that performs the query.
 * @return A ConstraintQueryResponse if successful, or an error.
 */
template <class Callable>
llvm::Expected<::ttnn::graph::ConstraintQueryResponse>
executeConstraintQuery(Callable &callable) {
  ::ttnn::graph::ConstraintQueryResponse query;
  try {
    query = callable();
  } catch (const std::exception &e) {
    // We expect that query will handle exceptions and set error message. If
    // not, we should not continue.
    // TODO(rpavlovicTT): This should be a TT_FATAL.
    llvm::errs() << "Exception thrown during op constraints query: " << e.what()
                 << "\n";
    assert(false && "Exception thrown during op constraints query");
  }

  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Op constraint query failed with error: " +
            query.error_message.value_or("<error message not set>"));
  }

  if (!query.output_tensor_spec.has_value()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Op constraint query missing output tensor");
  }

  return query;
}

/**
 * @brief Retrieves operation constraints based on the provided operation name
 * and callable.
 *
 * This function attempts to query operation constraints using the provided
 * callable and arguments. If successful, it returns a tuple with resource usage
 * details and the actual layout of the output tensor of the op. Otherwise, an
 * error message.
 *
 * @param name The name of the operation to query constraints for.
 * @param context The MLIRContext to use for creating the TTNNLayoutAttr for the
 * output tensor
 * @param deviceGrid The worker grid of the device the op is targetted for.
 * Required for creating the output tensor layout
 * @param callable A callable object that performs the query.
 * @return A tuple containing query results or a string error.
 */
template <class Callable>
llvm::Expected<OpConstraints> getOpConstraints(MLIRContext *context,
                                               ttcore::GridAttr deviceGrid,
                                               Callable &callable) {

  llvm::Expected<::ttnn::graph::ConstraintQueryResponse> query =
      executeConstraintQuery<Callable>(callable);
  if (auto error = query.takeError()) {
    return error;
  }

  ::ttnn::graph::ConstraintQueryResponse response = query.get();

  return OpConstraints(
      response.resource_usage.cb_peak_size_per_core,
      response.resource_usage.l1_buffers_peak_per_core,
      response.resource_usage.peak_memory_usage_per_core,
      response.resource_usage.l1_output_buffer_per_core,
      conversion::getLayoutAttrFromTensorSpec(
          context, response.output_tensor_spec.value(), deviceGrid.getShape()));
}

template <class Callable>
llvm::Expected<size_t> getOpRuntime(Callable &callable) {
  ::ttnn::graph::RuntimeQueryResponse query;
  try {
    query = callable();
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // Check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        query.error_message.value_or("<error message not set>"));
  }

  return query.runtime;
}

} // namespace operation

namespace detail {
/**
 * @brief Checks the validity of the compute grid size.
 *
 * This function verifies the dimensions and properties of the provided compute
 * grid size.
 *
 * @param computeGridSize The size of the compute grid, represented as a
 * CoreCoord object.
 * @param workerGrid The worker grid attributes, represented as a GridAttr
 * object. The shape of the worker grid is expected to be in the format {y, x}.
 *
 * @throws std::runtime_error If the worker grid size does not match the compute
 * grid size.
 */
void checkGrid(const ::tt::tt_metal::CoreCoord &computeGridSize,
               ttcore::GridAttr workerGrid) {
  // metal CoreCoord holds x,y
  // GridAttr holds shape {y,x}
  if ((static_cast<size_t>(workerGrid.getShape()[1]) != computeGridSize.x) ||
      (static_cast<size_t>(workerGrid.getShape()[0]) != computeGridSize.y)) {
    throw std::runtime_error("Selected worker grid is different than available "
                             "grid size. Compute Grid Size: " +
                             computeGridSize.str() + ", Worker Grid Size: (x=" +
                             std::to_string(workerGrid.getShape()[1]) + ",y=" +
                             std::to_string(workerGrid.getShape()[0]) + ")");
  }
}

/**
 * @brief Convenience wrapper to create and validate a tensor spec
 *
 * @param device Pointer to an open device to obtain the compute grid size
 */
llvm::Expected<::ttnn::TensorSpec>
convertToTensorSpec(::tt::tt_metal::distributed::MeshDevice *device,
                    llvm::ArrayRef<int64_t> shape, TTNNLayoutAttr layout) {
  const ::ttnn::TensorSpec spec = conversion::getTensorSpec(shape, layout);
  if (conversion::validateTensorSpec(
          spec, device->compute_with_storage_grid_size())) {
    return spec;
  }

  return llvm::createStringError(
      "Unable to create TensorSpec out of given shape and layout");
}

std::optional<::ttnn::TensorSpec>
convertToOptionalTensorSpec(::tt::tt_metal::distributed::MeshDevice *device,
                            std::optional<llvm::ArrayRef<int64_t>> shape,
                            std::optional<TTNNLayoutAttr> layout) {
  std::optional<::ttnn::TensorSpec> ret = std::nullopt;
  if (shape.has_value() && layout.has_value()) {
    auto retExp =
        detail::convertToTensorSpec(device, shape.value(), layout.value());
    if (!retExp) {
      assert(false && "Failed to convert to TensorSpec");
      return std::nullopt;
    }
    ret = retExp.get();
  }
  return ret;
}

/**
 * @brief Convenience wrapper to get a memory config from a TTNNLayout attr that
 * may be a nullptr. Returns std::nullopt if layout is nullptr
 */
std::optional<::tt::tt_metal::MemoryConfig>
getNullableMemoryConfig(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getMemoryConfig(layout);
}

/**
 * @brief Convenience wrapper to get a DataType from a TTNNLayout attr that
 * may be a nullptr. Returns std::nullopt if layout is nullptr
 */
std::optional<::tt::tt_metal::DataType>
getNullableDataType(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getDataType(layout.getDataType());
}

/**
 * @brief Checks if a C++ type T is compatible with a given MLIR type.
 *
 * @param elType The type to check.
 * @return True if the type is compatible, false otherwise.
 */
template <typename T>
bool isCompatibleType(mlir::Type elType) {
  if constexpr (std::is_same_v<T, float>) {
    return elType.isF32();
  } else if constexpr (std::is_same_v<T, double>) {
    return elType.isF64();
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return elType.isInteger(8);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return elType.isInteger(16);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return elType.isInteger(32);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return elType.isInteger(64);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return elType.isUnsignedInteger(8);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return elType.isUnsignedInteger(16);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return elType.isUnsignedInteger(32);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return elType.isUnsignedInteger(64);
  }
  return false;
}

/**
 * @brief This function populates a vector with the data that the ElementsAttr
 * contains. It checks for type compatibility and handles DenseElementsAttr and
 * SplatElementsAttr.
 */
template <typename T>
llvm::Expected<std::vector<T>>
getRawDataFromElementsAttr(mlir::ElementsAttr attr) {
  std::vector<T> result;
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!isCompatibleType<T>(denseAttr.getType().getElementType())) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch");
    }
    // Iterate over the elements
    for (auto value : denseAttr.getValues<T>()) {
      result.push_back(value);
    }

  } else if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(attr)) {
    // Handle splat attributes, Although this is not expected to be triggered
    // (since we have other ops to cover splat attributes, such as FullOp,
    // EmptyOp, etc), we can handle it here to avoid unnecessary failures.
    if (!isCompatibleType<T>(splatAttr.getType().getElementType())) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch");
    }
    auto splatValue = splatAttr.getSplatValue<T>();
    auto numElements = splatAttr.getType().getNumElements();
    result.resize(numElements, splatValue);
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unsupported ElementsAttr type");
  }
  return result;
}

// Template specialization for bfloat16 - MLIR doesn't have built-in support
// for bfloat16 in DenseElementsAttr::getValues<T>(), so we extract as uint16_t
// and convert to bfloat16
template <>
llvm::Expected<std::vector<bfloat16>>
getRawDataFromElementsAttr<bfloat16>(mlir::ElementsAttr attr) {
  std::vector<bfloat16> result;
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!denseAttr.getType().getElementType().isBF16()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch - expected BF16");
    }
    // Extract raw bytes as uint16_t and convert to bfloat16
    for (auto value : denseAttr.getValues<llvm::APFloat>()) {
      uint16_t rawBits =
          static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
      result.emplace_back(rawBits);
    }
  } else if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(attr)) {
    if (!splatAttr.getType().getElementType().isBF16()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch - expected BF16");
    }
    auto splatValue = splatAttr.getSplatValue<llvm::APFloat>();
    uint16_t rawBits =
        static_cast<uint16_t>(splatValue.bitcastToAPInt().getZExtValue());
    auto numElements = splatAttr.getType().getNumElements();
    result.resize(numElements, bfloat16(rawBits));
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unsupported ElementsAttr type");
  }
  return result;
}

template <typename OpTy>
auto getOpSymbol() {
  if constexpr (std::is_same_v<OpTy, ReluOp>) {
    return ::ttnn::relu;
  } else if constexpr (std::is_same_v<OpTy, Relu6Op>) {
    return ::ttnn::relu6;
  } else if constexpr (std::is_same_v<OpTy, SqrtOp>) {
    return ::ttnn::sqrt;
  } else if constexpr (std::is_same_v<OpTy, SinOp>) {
    return ::ttnn::sin;
  } else if constexpr (std::is_same_v<OpTy, AbsOp>) {
    return ::ttnn::abs;
  } else if constexpr (std::is_same_v<OpTy, CeilOp>) {
    return ::ttnn::ceil;
  } else if constexpr (std::is_same_v<OpTy, SignOp>) {
    return ::ttnn::sign;
  } else if constexpr (std::is_same_v<OpTy, FloorOp>) {
    return ::ttnn::floor;
  } else if constexpr (std::is_same_v<OpTy, IsFiniteOp>) {
    return ::ttnn::isfinite;
  } else if constexpr (std::is_same_v<OpTy, ExpOp>) {
    return ::ttnn::exp;
  } else if constexpr (std::is_same_v<OpTy, ErfOp>) {
    return ::ttnn::erf;
  } else if constexpr (std::is_same_v<OpTy, ErfcOp>) {
    return ::ttnn::erfc;
  } else if constexpr (std::is_same_v<OpTy, GeluOp>) {
    return ::ttnn::gelu;
  } else if constexpr (std::is_same_v<OpTy, RsqrtOp>) {
    return ::ttnn::rsqrt;
  } else if constexpr (std::is_same_v<OpTy, LogicalNotOp>) {
    return ::ttnn::logical_not;
  } else if constexpr (std::is_same_v<OpTy, NegOp>) {
    return ::ttnn::neg;
  } else if constexpr (std::is_same_v<OpTy, TanOp>) {
    return ::ttnn::tan;
  } else if constexpr (std::is_same_v<OpTy, AtanOp>) {
    return ::ttnn::atan;
  } else if constexpr (std::is_same_v<OpTy, Log1pOp>) {
    return ::ttnn::log1p;
  } else if constexpr (std::is_same_v<OpTy, Expm1Op>) {
    return ::ttnn::expm1;
  } else if constexpr (std::is_same_v<OpTy, CosOp>) {
    return ::ttnn::cos;
  } else if constexpr (std::is_same_v<OpTy, TanhOp>) {
    return ::ttnn::tanh;
  } else if constexpr (std::is_same_v<OpTy, LogOp>) {
    return ::ttnn::log;
  } else if constexpr (std::is_same_v<OpTy, ReciprocalOp>) {
    return ::ttnn::reciprocal;
  } else if constexpr (std::is_same_v<OpTy, CbrtOp>) {
    return ::ttnn::cbrt;
  } else if constexpr (std::is_same_v<OpTy, BitwiseNotOp>) {
    return ::ttnn::bitwise_not;
  } else if constexpr (std::is_same_v<OpTy, AddOp>) {
    return ::ttnn::add;
  } else if constexpr (std::is_same_v<OpTy, MultiplyOp>) {
    return ::ttnn::multiply;
  } else if constexpr (std::is_same_v<OpTy, SubtractOp>) {
    return ::ttnn::subtract;
  } else if constexpr (std::is_same_v<OpTy, LogicalRightShiftOp>) {
    return ::ttnn::logical_right_shift;
  } else if constexpr (std::is_same_v<OpTy, LogicalLeftShiftOp>) {
    return ::ttnn::logical_left_shift;
  } else if constexpr (std::is_same_v<OpTy, DivideOp>) {
    return ::ttnn::divide;
  } else if constexpr (std::is_same_v<OpTy, EqualOp>) {
    return ::ttnn::eq;
  } else if constexpr (std::is_same_v<OpTy, NotEqualOp>) {
    return ::ttnn::ne;
  } else if constexpr (std::is_same_v<OpTy, GreaterEqualOp>) {
    return ::ttnn::ge;
  } else if constexpr (std::is_same_v<OpTy, GreaterThanOp>) {
    return ::ttnn::gt;
  } else if constexpr (std::is_same_v<OpTy, LessEqualOp>) {
    return ::ttnn::le;
  } else if constexpr (std::is_same_v<OpTy, LessThanOp>) {
    return ::ttnn::lt;
  } else if constexpr (std::is_same_v<OpTy, LogicalAndOp>) {
    return ::ttnn::logical_and;
  } else if constexpr (std::is_same_v<OpTy, LogicalOrOp>) {
    return ::ttnn::logical_or;
  } else if constexpr (std::is_same_v<OpTy, LogicalXorOp>) {
    return ::ttnn::logical_xor;
  } else if constexpr (std::is_same_v<OpTy, MaximumOp>) {
    return ::ttnn::maximum;
  } else if constexpr (std::is_same_v<OpTy, MinimumOp>) {
    return ::ttnn::minimum;
  } else if constexpr (std::is_same_v<OpTy, BitwiseAndOp>) {
    return ::ttnn::bitwise_and;
  } else if constexpr (std::is_same_v<OpTy, BitwiseOrOp>) {
    return ::ttnn::bitwise_or;
  } else if constexpr (std::is_same_v<OpTy, BitwiseXorOp>) {
    return ::ttnn::bitwise_xor;
  } else if constexpr (std::is_same_v<OpTy, RemainderOp>) {
    return ::ttnn::remainder;
  } else if constexpr (std::is_same_v<OpTy, Atan2Op>) {
    return ::ttnn::atan2;
  } else if constexpr (std::is_same_v<OpTy, PowOp>) {
    return ::ttnn::pow;
  } else if constexpr (std::is_same_v<OpTy, WhereOp>) {
    return ::ttnn::where;
  } else if constexpr (std::is_same_v<OpTy, MeanOp>) {
    return ::ttnn::mean;
  } else if constexpr (std::is_same_v<OpTy, MaxOp>) {
    return ::ttnn::max;
  } else if constexpr (std::is_same_v<OpTy, MinOp>) {
    return ::ttnn::min;
  } else if constexpr (std::is_same_v<OpTy, SumOp>) {
    return ::ttnn::sum;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ZerosOp>) {
    return ::ttnn::zeros;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::OnesOp>) {
    return ::ttnn::ones;
  } else if constexpr (std::is_same_v<OpTy, QuantizeOp>) {
    return ::ttnn::quantize;
  } else if constexpr (std::is_same_v<OpTy, DequantizeOp>) {
    return ::ttnn::dequantize;
  } else if constexpr (std::is_same_v<OpTy, GlobalAvgPool2dOp>) {
    return ::ttnn::global_avg_pool2d;
  } else {
    static_assert(ttmlir::utils::always_false(),
                  "add mapping from TTNN dialect to TTNN lib op");
  }
}

} // namespace detail
#endif // TTMLIR_ENABLE_OPMODEL

bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 TTNNLayoutAttr layout,
                                 ttcore::GridAttr maxGrid) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Conversion to TensorSpec may throw if the layout is invalid, in which case
  // we return false.
  try {
    auto tensorSpec = conversion::getTensorSpec(tensorShape, layout);
    auto computeGridSize = ::tt::tt_metal::CoreCoord{
        static_cast<std::size_t>(maxGrid.getShape()[0]),
        static_cast<std::size_t>(maxGrid.getShape()[1])};
    return conversion::validateTensorSpec(tensorSpec, computeGridSize);
  } catch (const std::exception &e) {
    return false;
  }
  return true;
#else
  return true;
#endif
}

#ifdef TTMLIR_ENABLE_OPMODEL

static ::tt::tt_metal::HostBuffer
createHostBuffer(uint32_t numElements, ::tt::tt_metal::DataType dataType) {
  switch (dataType) {
  case ::tt::tt_metal::DataType::FLOAT32: {
    std::vector<float> data(numElements);
    return ::tt::tt_metal::HostBuffer(std::move(data));
  }
  case ::tt::tt_metal::DataType::BFLOAT16: {
    std::vector<bfloat16> data(numElements);
    return ::tt::tt_metal::HostBuffer(std::move(data));
  }
  default:
    llvm::report_fatal_error("Unsupported data type");
  }
}

// Allocate a ttnn tensor with the given shape and data type.
static ::tt::tt_metal::Tensor
createMetalHostTensor(llvm::ArrayRef<int64_t> shape,
                      ttcore::DataType dataType) {
  // Calculate total volume of the tensor
  uint32_t volume = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    volume *= shape[i];
  }

  auto metalDataType = conversion::getDataType(dataType);
  auto hostBuffer = createHostBuffer(volume, metalDataType);
  auto metalShape = conversion::getShape(shape);
  ::tt::tt_metal::PageConfig pageconfig(::tt::tt_metal::Layout::ROW_MAJOR);
  ::tt::tt_metal::TensorLayout layout(metalDataType, pageconfig,
                                      ::tt::tt_metal::MemoryConfig{});
  ::tt::tt_metal::TensorSpec tensorSpec(metalShape, layout);

  return ::tt::tt_metal::Tensor(std::move(hostBuffer), tensorSpec);
}

// Returns the output tensor spec of the prepared weights for a conv2d op.
// Transform the standard OIHW weights layout to the ttnn convolution internal
// layout that is desired. The output shape is dependant on the conv2d config
// and input memory config.
static llvm::Expected<::ttnn::TensorSpec>
getPrepareConv2dWeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig, bool hasBias,
    bool transpose) {
  if (weightLayout.getBufferType() != BufferType::SystemMemory) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d weight tensor assumed to be on host.");
  }
  if (weightLayout.getDataType() != ttcore::DataType::Float32 &&
      weightLayout.getDataType() != ttcore::DataType::BFloat16) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d weight tensor assumed to be float32 or bfloat16.");
  }

  // TODO(rpavlovicTT):: Move this to tt-metal side #4043
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::DataType> inputDtype =
      detail::getNullableDataType(inputLayout);

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(weightLayout);

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto prepareConv2dWeightsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv2d::prepare_conv_weights, device,
        weightTensor, inputSpec.memory_config(), inputSpec.layout(), "OIHW",
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, *inputDtype, outputDtype,
        conv2dConfigConverted,
        /* compute_config_ */ std::nullopt,
        /* dram_slice_config_ */ std::nullopt);
  };

  auto prepareConvTranspose2dWeightsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv_transpose2d::
            prepare_conv_transpose2d_weights,
        device, weightTensor, inputSpec.memory_config(), inputSpec.layout(),
        "IOHW", in_channels, out_channels, batch_size, input_height,
        input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, *inputDtype, outputDtype,
        conv2dConfigConverted,
        /* compute_config_ */ std::nullopt,
        /* mirror_kernel */ true);
  };

  auto output =
      transpose
          ? operation::executeConstraintQuery(
                prepareConvTranspose2dWeightsOpQuery)
          : operation::executeConstraintQuery(prepareConv2dWeightsOpQuery);

  if (!output) {
    return output.takeError();
  }

  assert(output.get().output_tensor_spec.has_value());
  return output.get().output_tensor_spec.value();
}

// Returns the output tensor spec of the prepared bias for a conv2d op.
static llvm::Expected<::ttnn::TensorSpec>
getPrepareConv2dBiasOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> biasShape, TTNNLayoutAttr biasLayout,
    ::tt::tt_metal::DataType weightsDtype, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig) {
  if (biasLayout.getBufferType() != BufferType::SystemMemory) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Conv2d bias tensor assumed to be on host.");
  }

  // TODO(rpavlovicTT):: Move this to tt-metal side #4043
  if (biasLayout.getDataType() != ttcore::DataType::Float32 &&
      biasLayout.getDataType() != ttcore::DataType::BFloat16) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d bias tensor assumed to be float32 or bfloat16.");
  }
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::DataType> inputDtype =
      detail::getNullableDataType(inputLayout);

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(biasLayout);

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  auto prepare_fn = &::ttnn::operations::conv::conv2d::prepare_conv_bias;
  // Create query closure
  auto prepareConv2dBiasOpQuery = [=]() {
    ::ttnn::operations::conv::conv2d::Conv2dConfig localConfig;
    if (!conv2dConfigConverted.has_value()) {
      localConfig = ::ttnn::operations::conv::conv2d::Conv2dConfig();
      // Weights dtype needs to be set for prepare_conv_bias.
      localConfig.weights_dtype = weightsDtype;
    } else {
      localConfig = *conv2dConfigConverted;
    }

    return ::ttnn::graph::query_op_constraints(
        prepare_fn, device, biasTensor, inputSpec.memory_config(),
        inputSpec.layout(), in_channels, out_channels, batch_size, input_height,
        input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, device, *inputDtype, outputDtype, localConfig,
        /*compute_config_=*/std::nullopt);
  };

  auto output = operation::executeConstraintQuery(prepareConv2dBiasOpQuery);

  if (!output) {
    return output.takeError();
  }

  assert(output.get().output_tensor_spec.has_value());
  return output.get().output_tensor_spec.value();
}

#endif // TTMLIR_ENABLE_OPMODEL

mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(Conv2dOp *op,
                                     Conv2dConfigAttr conv2dConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto input = op->getInput().getType();
  auto weight = op->getWeight().getType();
  auto inputLayout = mlir::cast<TTNNLayoutAttr>(input.getEncoding());
  auto weightLayout = mlir::cast<TTNNLayoutAttr>(weight.getEncoding());

  llvm::Expected<::ttnn::TensorSpec> outputTensorSpec =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          input.getShape(), inputLayout, weight.getShape(), weightLayout,
          op->getInChannels(), op->getOutChannels(), op->getBatchSize(),
          op->getInputHeight(), op->getInputWidth(), op->getKernelSize(),
          op->getStride(), op->getPadding(), op->getDilation(), op->getGroups(),
          conv2dConfig, op->getBias() != nullptr,
          /* transpose */ false);

  if (!outputTensorSpec) {
    llvm::errs() << llvm::toString(outputTensorSpec.takeError());
    assert(false && "Failed to calculate conv2d prepared weights shape.");
  }

  // Convert back to RankedTensorType
  auto deviceGrid =
      ttcore::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

  auto outputLayout = conversion::getLayoutAttrFromTensorSpec(
      op->getContext(), outputTensorSpec.get(), deviceGrid);

  auto shape = outputTensorSpec.get().logical_shape();

  return mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend()),
      outputLayout.getScalarElementType(), outputLayout);
#else
  assert(false &&
         "Cannot calculate conv2d prepared weights shape without op model");
#endif
}

//===----------------------------------------------------------------------===//
// Device
//===----------------------------------------------------------------------===//

llvm::Expected<bool> Device::getDeviceConstraints(ttcore::GridAttr workerGrid) {
#ifdef TTMLIR_ENABLE_OPMODEL
  try {
    detail::checkGrid(SingletonDeviceContext::getInstance()
                          .getDevice()
                          ->compute_with_storage_grid_size(),
                      workerGrid);
    return true;
  } catch (const std::exception &e) {
    return llvm::createStringError(e.what());
  }
#endif
  return true;
}

//===----------------------------------------------------------------------===//
// Unary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> UnaryEltwiseOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t>
UnaryEltwiseOpModel<OpTy>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                        TTNNLayoutAttr inputLayout,
                                        TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryEltwiseWithFastApproxModeOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  bool fastApproxMode = true;

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device, inputSpec, fastApproxMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t>
UnaryEltwiseWithFastApproxModeOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  bool fastApproxMode = true;

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec, fastApproxMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for UnaryEltwiseOpModel.
template struct UnaryEltwiseOpModel<ReluOp>;
template struct UnaryEltwiseOpModel<Relu6Op>;
template struct UnaryEltwiseOpModel<SqrtOp>;
template struct UnaryEltwiseOpModel<SinOp>;
template struct UnaryEltwiseOpModel<AbsOp>;
template struct UnaryEltwiseOpModel<CosOp>;
template struct UnaryEltwiseOpModel<TanhOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<LogOp>;
template struct UnaryEltwiseOpModel<CeilOp>;
template struct UnaryEltwiseOpModel<SignOp>;
template struct UnaryEltwiseOpModel<FloorOp>;
template struct UnaryEltwiseOpModel<IsFiniteOp>;
template struct UnaryEltwiseOpModel<LogicalNotOp>;
template struct UnaryEltwiseOpModel<NegOp>;
template struct UnaryEltwiseOpModel<TanOp>;
template struct UnaryEltwiseOpModel<AtanOp>;
template struct UnaryEltwiseOpModel<ReciprocalOp>;
template struct UnaryEltwiseOpModel<CbrtOp>;
template struct UnaryEltwiseOpModel<BitwiseNotOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<Log1pOp>;
template struct UnaryEltwiseOpModel<Expm1Op>;
template struct UnaryEltwiseOpModel<RsqrtOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ErfOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ErfcOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ExpOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<GeluOp>;

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<SigmoidOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Add default parameters
  int32_t vectorMode =
      static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
  bool approximateMode = false;

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::sigmoid, device, inputSpec, vectorMode, approximateMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<SigmoidOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                 TTNNLayoutAttr inputLayout,
                                 TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Add default parameters
  int32_t vectorMode =
      static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
  bool approximateMode = false;

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::sigmoid, device, inputSpec, vectorMode, approximateMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<LeakyReluOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::APFloat slope,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto leakyReluOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::leaky_relu, device, inputSpec, slope.convertToFloat(),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     leakyReluOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LeakyReluOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat slope, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto leakyReluOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::leaky_relu, device, inputSpec, slope.convertToFloat(),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(leakyReluOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Binary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryEltwiseOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
    TTNNLayoutAttr inputLayoutB, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(detail::getOpSymbol<OpTy>(),
                                               device, inputSpecA, inputSpecB,
                                               outputDType, outputMemoryConfig);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> BinaryEltwiseOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(detail::getOpSymbol<OpTy>(), device,
                                           inputSpecA, inputSpecB, outputDType,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryCompositeOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
    TTNNLayoutAttr inputLayoutB, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(detail::getOpSymbol<OpTy>(),
                                               device, inputSpecA, inputSpecB,
                                               outputMemoryConfig);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> BinaryCompositeOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(detail::getOpSymbol<OpTy>(), device,
                                           inputSpecA, inputSpecB,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for BinaryEltwiseOpModel.
template struct BinaryEltwiseOpModel<AddOp>;
template struct BinaryEltwiseOpModel<MultiplyOp>;
template struct BinaryEltwiseOpModel<LogicalRightShiftOp>;
template struct BinaryEltwiseOpModel<SubtractOp>;
template struct BinaryEltwiseOpModel<MaximumOp>;
template struct BinaryEltwiseOpModel<MinimumOp>;
template struct BinaryEltwiseOpModel<DivideOp>;
template struct BinaryEltwiseOpModel<EqualOp>;
template struct BinaryEltwiseOpModel<NotEqualOp>;
template struct BinaryEltwiseOpModel<GreaterEqualOp>;
template struct BinaryEltwiseOpModel<GreaterThanOp>;
template struct BinaryEltwiseOpModel<LessEqualOp>;
template struct BinaryEltwiseOpModel<LessThanOp>;
template struct BinaryEltwiseOpModel<LogicalAndOp>;
template struct BinaryEltwiseOpModel<LogicalOrOp>;
template struct BinaryEltwiseOpModel<LogicalXorOp>;
template struct BinaryEltwiseOpModel<PowOp>;
// BinaryCompositeOpModel
template struct BinaryCompositeOpModel<BitwiseAndOp>;
template struct BinaryCompositeOpModel<BitwiseOrOp>;
template struct BinaryCompositeOpModel<BitwiseXorOp>;
template struct BinaryCompositeOpModel<LogicalLeftShiftOp>;
template struct BinaryCompositeOpModel<RemainderOp>;
template struct BinaryCompositeOpModel<Atan2Op>;

//===----------------------------------------------------------------------===//
// Ternary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> TernaryEltwiseOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
    TTNNLayoutAttr inputLayoutB, llvm::ArrayRef<int64_t> inputShapeC,
    TTNNLayoutAttr inputLayoutC, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  auto inputSpecCExp =
      detail::convertToTensorSpec(device, inputShapeC, inputLayoutC);
  if (!inputSpecCExp) {
    return inputSpecCExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecC = inputSpecCExp.get();

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(detail::getOpSymbol<OpTy>(),
                                               device, inputSpecA, inputSpecB,
                                               inputSpecC, outputMemoryConfig);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> TernaryEltwiseOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC, TTNNLayoutAttr inputLayoutC,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  auto inputSpecCExp =
      detail::convertToTensorSpec(device, inputShapeC, inputLayoutC);
  if (!inputSpecCExp) {
    return inputSpecCExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecC = inputSpecCExp.get();

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(detail::getOpSymbol<OpTy>(), device,
                                           inputSpecA, inputSpecB, inputSpecC,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for TernaryEltwiseOpModel.
template struct TernaryEltwiseOpModel<WhereOp>;

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> ReductionOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, std::optional<llvm::ArrayRef<int64_t>> dimArg,
    bool keepDim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::ttsl::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device, inputSpec, dimArgConverted,
        keepDim, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> ReductionOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::ttsl::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec, dimArgConverted,
        keepDim, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for ReductionOpModel.
template struct ReductionOpModel<MeanOp>;
template struct ReductionOpModel<SumOp>;
template struct ReductionOpModel<MaxOp>;
template struct ReductionOpModel<MinOp>;

//===----------------------------------------------------------------------===//
// Named Full Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> NamedFullOpModel<OpTy>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, mlir::tt::ttnn::ShapeAttr shape,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    std::optional<mlir::tt::ttnn::Layout> layout,
    std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  std::optional<::tt::tt_metal::DataType> metalDtype = std::nullopt;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }
  std::optional<::ttnn::Layout> metalLayout = std::nullopt;
  if (layout.has_value()) {
    metalLayout = conversion::getPageLayout(layout.value());
  }
  std::optional<::ttnn::MemoryConfig> metalMemoryConfig = std::nullopt;
  if (outputLayout) {
    metalMemoryConfig = conversion::getMemoryConfig(outputLayout);
  } else if (memoryConfig.has_value()) {
    metalMemoryConfig = conversion::getMemoryConfig(memoryConfig.value());
  }
  std::optional<std::reference_wrapper<::tt::tt_metal::distributed::MeshDevice>>
      deviceRef = *device;

  auto namedFullOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device,
        conversion::getShape(shape.getShape()), metalDtype, metalLayout,
        deviceRef, metalMemoryConfig);
  };
  return operation::getOpConstraints(shape.getContext(), deviceGrid,
                                     namedFullOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for NamedFullOpModel.
template struct NamedFullOpModel<ZerosOp>;
template struct NamedFullOpModel<OnesOp>;

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<SoftmaxOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, const int dimArg, bool numericStable,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto softmaxOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::softmax, device, inputSpec, dimArg,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt, // compute_kernel_config,
        numericStable);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     softmaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SoftmaxOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dimArg, bool numericStable, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto softmaxOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::softmax, device, inputSpec, dimArg,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt, // compute_kernel_config,
        numericStable);
  };

  return operation::getOpRuntime(softmaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ReshapeOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> outputShape,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto reshapeOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::reshape, device, inputSpec, conversion::getShape(outputShape),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     reshapeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ReshapeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto reshapeOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::reshape, device, inputSpec, conversion::getShape(outputShape),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(reshapeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SliceStaticOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<SliceStaticOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> begins,
    llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // convert arrays
  ::ttsl::SmallVector<int> beginsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(begins);
  ::ttsl::SmallVector<int> endsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(ends);
  ::ttsl::SmallVector<int> stepVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(step);

  ttsl::Span<const int> beginsSpan = ::ttsl::make_const_span(beginsVec);
  ttsl::Span<const int> endsSpan = ::ttsl::make_const_span(endsVec);
  ttsl::Span<const int> stepSpan = ::ttsl::make_const_span(stepVec);

  // Create query closure
  auto sliceOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::slice, device, inputSpec, beginsSpan, endsSpan, stepSpan,
        detail::getNullableMemoryConfig(outputLayout), std::nullopt,
        std::nullopt);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     sliceOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SliceStaticOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
    llvm::ArrayRef<int64_t> step, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Convert arrays
  ::ttsl::SmallVector<int> beginsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(begins);
  ::ttsl::SmallVector<int> endsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(ends);
  ::ttsl::SmallVector<int> stepVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(step);

  ttsl::Span<const int> beginsSpan = ::ttsl::make_const_span(beginsVec);
  ttsl::Span<const int> endsSpan = ::ttsl::make_const_span(endsVec);
  ttsl::Span<const int> stepSpan = ::ttsl::make_const_span(stepVec);

  // Create query closure
  auto sliceOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::slice, device, inputSpec, beginsSpan, endsSpan, stepSpan,
        detail::getNullableMemoryConfig(outputLayout), std::nullopt,
        std::nullopt);
  };

  return operation::getOpRuntime(sliceOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SliceDynamicOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<SliceDynamicOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> beginsShape,
    TTNNLayoutAttr beginsLayout, llvm::ArrayRef<int64_t> endsShape,
    TTNNLayoutAttr endsLayout, std::optional<llvm::SmallVector<int64_t>> step,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // It is not possible to use the dynamic version of slice in tt-metal since
  // the validity of the op depends on the actual data that is stored in the
  // begins/ends tensors (which is not available at compile time). Therefore,
  // here we approximate the op by using the static version and calling
  // (possibly) the worst case scenario for the static version which is slicing
  // from the beginning to one index before the end (Capturing the entire tensor
  // except for one row results in the highest memory usage). Note that this is
  // a fairly accurate approximation since the dynamic version in metal also
  // converts the three tensors (begins, ends, step) to vectors and then calls
  // the static version.
  ::ttsl::SmallVector<int> stepVec(inputShape.size(), 1);
  ::ttsl::SmallVector<int> beginsVec(inputShape.size(), 0);
  ::ttsl::SmallVector<int> endsVec(inputShape.begin(), inputShape.end());
  std::ranges::for_each(endsVec, [](int &end) { end = end - 1; });

  // Default values in tt-metal:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<float> padValue = std::nullopt;
  // Create query closure to make a call to the static version of the op:
  auto sliceOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::slice, device, inputSpec, beginsVec, endsVec, stepVec,
        detail::getNullableMemoryConfig(outputLayout), outputSpec, padValue);
  };
  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     sliceOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SliceDynamicOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> beginsShape, TTNNLayoutAttr beginsLayout,
    llvm::ArrayRef<int64_t> endsShape, TTNNLayoutAttr endsLayout,
    std::optional<llvm::SmallVector<int64_t>> step,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // It is not possible to use the dynamic version of slice in tt-metal since
  // the validity of the op depends on the actual data that is stored in the
  // begins/ends tensors (which is not available at compile time). Therefore,
  // here we approximate the op by using the static version and calling
  // (possibly) the worst case scenario for the static version which is slicing
  // from the beginning to the end with a step of 2 (Capturing all possible
  // stripes of data from the input tensor is the most run time intensive
  // pattern).
  // Note that this is a fairly accurate approximation since the dynamic version
  // in metal also converts the three tensors (begins, ends, step) to vectors
  // and then calls the static version.
  ::ttsl::SmallVector<int> stepVec(inputShape.size(), 2);
  ::ttsl::SmallVector<int> beginsVec(inputShape.size(), 0);
  ::ttsl::SmallVector<int> endsVec(inputShape.begin(), inputShape.end());
  // Default values in tt-metal:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<float> padValue = std::nullopt;

  // Create query closure to make a call to the static version of the op:
  auto sliceOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::slice, device, inputSpec, beginsVec, endsVec, stepVec,
        detail::getNullableMemoryConfig(outputLayout), outputSpec, padValue);
  };

  return operation::getOpRuntime(sliceOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<TypecastOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, ttcore::DataTypeAttr dtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto typecastOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     typecastOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<TypecastOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto typecastOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(typecastOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ToLayoutOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, std::optional<ttcore::DataType> outputDtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype) {
    dtype = conversion::getDataType(outputDtype.value());
  } else {
    dtype = std::nullopt;
  }

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        detail::getNullableMemoryConfig(outputLayout));
  };
  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     toLayoutOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ToLayoutOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype) {
    dtype = conversion::getDataType(outputDtype.value());
  } else {
    dtype = std::nullopt;
  }

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(toLayoutOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ToMemoryConfigOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, MemoryConfigAttr memoryConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto toMemoryConfigOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::to_memory_config, device, inputSpec,
        conversion::getMemoryConfig(memoryConfig));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     toMemoryConfigOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ToMemoryConfigOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    MemoryConfigAttr memoryConfig, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto toMemoryConfigOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::to_memory_config, device, inputSpec,
        conversion::getMemoryConfig(memoryConfig));
  };

  return operation::getOpRuntime(toMemoryConfigOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ConcatOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid,
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  assert(inputShapes.size() == inputLayouts.size());
  size_t numInputs = inputShapes.size();

  std::vector<::ttnn::TensorSpec> inputSpecs;
  for (size_t i = 0; i < numInputs; ++i) {
    auto inputSpecExp =
        detail::convertToTensorSpec(device, inputShapes[i], inputLayouts[i]);
    if (!inputSpecExp) {
      return inputSpecExp.takeError();
    }
    inputSpecs.push_back(inputSpecExp.get());
  }

  // Create query closure
  auto concatOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::concat, device, inputSpecs, dim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayouts[0].getContext(), deviceGrid,
                                     concatOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ConcatOp>::getOpRuntime(
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  assert(inputShapes.size() == inputLayouts.size());
  size_t numInputs = inputShapes.size();

  std::vector<::ttnn::TensorSpec> inputSpecs;
  for (size_t i = 0; i < numInputs; ++i) {
    auto inputSpecExp =
        detail::convertToTensorSpec(device, inputShapes[i], inputLayouts[i]);
    if (!inputSpecExp) {
      return inputSpecExp.takeError();
    }
    inputSpecs.push_back(inputSpecExp.get());
  }

  // Create query closure
  auto concatOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::concat, device, inputSpecs, dim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(concatOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<TransposeOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto transposeOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::transpose, device, inputSpec, dim0, dim1,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     transposeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<TransposeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dim0, const int dim1, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto transposeOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::transpose, device, inputSpec, dim0, dim1,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(transposeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MorehCumSumOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<MorehCumSumOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, const int64_t dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto morehCumSumOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::moreh_cumsum, device, inputSpec, dim, std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     morehCumSumOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MorehCumSumOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int64_t dim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto morehCumSumOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::moreh_cumsum, device, inputSpec, dim, std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(morehCumSumOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<ConcatenateHeadsOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto concatenateHeadsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::transformer::concatenate_heads, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     concatenateHeadsOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<ConcatenateHeadsOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                          TTNNLayoutAttr inputLayout,
                                          TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto concatenateHeadsOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::transformer::concatenate_heads, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(concatenateHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<ScaledDotProductAttentionDecodeOp>::getOpConstraints(
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
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto querySpecExp =
      detail::convertToTensorSpec(device, queryShape, queryLayout);
  if (!querySpecExp) {
    return querySpecExp.takeError();
  }
  auto keySpecExp = detail::convertToTensorSpec(device, keyShape, keyLayout);
  if (!keySpecExp) {
    return keySpecExp.takeError();
  }
  auto valueSpecExp =
      detail::convertToTensorSpec(device, valueShape, valueLayout);
  if (!valueSpecExp) {
    return valueSpecExp.takeError();
  }
  auto curPosTensorSpecExp = detail::convertToTensorSpec(
      device, curPosTensorShape, curPosTensorLayout);
  if (!curPosTensorSpecExp) {
    return curPosTensorSpecExp.takeError();
  }

  ::ttnn::TensorSpec querySpec = querySpecExp.get();
  ::ttnn::TensorSpec keySpec = keySpecExp.get();
  ::ttnn::TensorSpec valueSpec = valueSpecExp.get();
  ::ttnn::TensorSpec curPosTensorSpec = curPosTensorSpecExp.get();

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  // The current position information is required for this op. It can either be
  // passed as a tensor or as a uint vector. The uint vector is not wrapped in a
  // std::optional so we must pass an empty vector.
  const std::vector<uint32_t> curPosEmpty = {};
  auto scaledDotProductAttentionDecodeOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::transformer::scaled_dot_product_attention_decode, device,
        querySpec, keySpec, valueSpec, isCausal, attentionMaskSpec, curPosEmpty,
        curPosTensorSpec, attentionSinkSpec, scaleFloat,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraints(queryLayout.getContext(), deviceGrid,
                                     scaledDotProductAttentionDecodeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ScaledDotProductAttentionDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    llvm::ArrayRef<int64_t> curPosTensorShape,
    TTNNLayoutAttr curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto querySpecExp =
      detail::convertToTensorSpec(device, queryShape, queryLayout);
  if (!querySpecExp) {
    return querySpecExp.takeError();
  }
  auto keySpecExp = detail::convertToTensorSpec(device, keyShape, keyLayout);
  if (!keySpecExp) {
    return keySpecExp.takeError();
  }
  auto valueSpecExp =
      detail::convertToTensorSpec(device, valueShape, valueLayout);
  if (!valueSpecExp) {
    return valueSpecExp.takeError();
  }
  auto curPosTensorSpecExp = detail::convertToTensorSpec(
      device, curPosTensorShape, curPosTensorLayout);
  if (!curPosTensorSpecExp) {
    return curPosTensorSpecExp.takeError();
  }

  ::ttnn::TensorSpec querySpec = querySpecExp.get();
  ::ttnn::TensorSpec keySpec = keySpecExp.get();
  ::ttnn::TensorSpec valueSpec = valueSpecExp.get();
  ::ttnn::TensorSpec curPosTensorSpec = curPosTensorSpecExp.get();

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);

  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;
  // The current position information is required for this op. It can either be
  // passed as a tensor or as a uint vector. The uint vector is not wrapped in a
  // std::optional so we must pass an empty vector.
  const std::vector<uint32_t> curPosEmpty = {};
  auto scaledDotProductAttentionDecodeOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::transformer::scaled_dot_product_attention_decode, device,
        querySpec, keySpec, valueSpec, isCausal, attentionMaskSpec, curPosEmpty,
        curPosTensorSpec, attentionSinkSpec, scaleFloat,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(scaledDotProductAttentionDecodeOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<ScaledDotProductAttentionOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> queryShape,
    TTNNLayoutAttr queryLayout, llvm::ArrayRef<int64_t> keyShape,
    TTNNLayoutAttr keyLayout, llvm::ArrayRef<int64_t> valueShape,
    TTNNLayoutAttr valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout, bool isCausal,
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto querySpecExp =
      detail::convertToTensorSpec(device, queryShape, queryLayout);
  if (!querySpecExp) {
    return querySpecExp.takeError();
  }
  auto keySpecExp = detail::convertToTensorSpec(device, keyShape, keyLayout);
  if (!keySpecExp) {
    return keySpecExp.takeError();
  }
  auto valueSpecExp =
      detail::convertToTensorSpec(device, valueShape, valueLayout);
  if (!valueSpecExp) {
    return valueSpecExp.takeError();
  }

  ::ttnn::TensorSpec querySpec = querySpecExp.get();
  ::ttnn::TensorSpec keySpec = keySpecExp.get();
  ::ttnn::TensorSpec valueSpec = valueSpecExp.get();

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  auto scaledDotProductAttentionOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::transformer::scaled_dot_product_attention, device, querySpec,
        keySpec, valueSpec, attentionMaskSpec, isCausal, scaleFloat,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraints(queryLayout.getContext(), deviceGrid,
                                     scaledDotProductAttentionOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ScaledDotProductAttentionOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout, bool isCausal,
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto querySpecExp =
      detail::convertToTensorSpec(device, queryShape, queryLayout);
  if (!querySpecExp) {
    return querySpecExp.takeError();
  }
  auto keySpecExp = detail::convertToTensorSpec(device, keyShape, keyLayout);
  if (!keySpecExp) {
    return keySpecExp.takeError();
  }
  auto valueSpecExp =
      detail::convertToTensorSpec(device, valueShape, valueLayout);
  if (!valueSpecExp) {
    return valueSpecExp.takeError();
  }

  ::ttnn::TensorSpec querySpec = querySpecExp.get();
  ::ttnn::TensorSpec keySpec = keySpecExp.get();
  ::ttnn::TensorSpec valueSpec = valueSpecExp.get();

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  auto scaledDotProductAttentionOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::transformer::scaled_dot_product_attention, device, querySpec,
        keySpec, valueSpec, attentionMaskSpec, isCausal, scaleFloat,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(scaledDotProductAttentionOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===-----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOp
// ===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<RotaryEmbeddingLlamaOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> cosShape,
    TTNNLayoutAttr cosLayout, llvm::ArrayRef<int64_t> sinShape,
    TTNNLayoutAttr sinLayout, llvm::ArrayRef<int64_t> transMatShape,
    TTNNLayoutAttr transMatLayout, bool isDecodeMode,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  auto cosSpecExp = detail::convertToTensorSpec(device, cosShape, cosLayout);
  if (!cosSpecExp) {
    return cosSpecExp.takeError();
  }
  auto sinSpecExp = detail::convertToTensorSpec(device, sinShape, sinLayout);
  if (!sinSpecExp) {
    return sinSpecExp.takeError();
  }
  auto transMatSpecExp =
      detail::convertToTensorSpec(device, transMatShape, transMatLayout);
  if (!transMatSpecExp) {
    return transMatSpecExp.takeError();
  }

  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();
  ::ttnn::TensorSpec cosSpec = cosSpecExp.get();
  ::ttnn::TensorSpec sinSpec = sinSpecExp.get();
  ::ttnn::TensorSpec transMatSpec = transMatSpecExp.get();

  auto rotaryEmbeddingLlamaOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::experimental::rotary_embedding_llama, device, inputSpec,
        cosSpec, sinSpec, transMatSpec, isDecodeMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     rotaryEmbeddingLlamaOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RotaryEmbeddingLlamaOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    llvm::ArrayRef<int64_t> transMatShape, TTNNLayoutAttr transMatLayout,
    bool isDecodeMode, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  auto cosSpecExp = detail::convertToTensorSpec(device, cosShape, cosLayout);
  if (!cosSpecExp) {
    return cosSpecExp.takeError();
  }
  auto sinSpecExp = detail::convertToTensorSpec(device, sinShape, sinLayout);
  if (!sinSpecExp) {
    return sinSpecExp.takeError();
  }
  auto transMatSpecExp =
      detail::convertToTensorSpec(device, transMatShape, transMatLayout);
  if (!transMatSpecExp) {
    return transMatSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();
  ::ttnn::TensorSpec cosSpec = cosSpecExp.get();
  ::ttnn::TensorSpec sinSpec = sinSpecExp.get();
  ::ttnn::TensorSpec transMatSpec = transMatSpecExp.get();

  // Create query closure
  auto rotaryEmbeddingLlamaOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::experimental::rotary_embedding_llama, device, inputSpec,
        cosSpec, sinSpec, transMatSpec, isDecodeMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(rotaryEmbeddingLlamaOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<NLPConcatHeadsOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto nlpConcatHeadsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::experimental::nlp_concat_heads, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     nlpConcatHeadsOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<NLPConcatHeadsOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                        TTNNLayoutAttr inputLayout,
                                        TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto nlpConcatHeadsOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::experimental::nlp_concat_heads, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(nlpConcatHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<NLPConcatHeadsDecodeOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, uint32_t numHeads,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto nlpConcatHeadsDecodeOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::experimental::nlp_concat_heads_decode, device, inputSpec,
        numHeads, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     nlpConcatHeadsDecodeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<NLPConcatHeadsDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t numHeads, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto nlpConcatHeadsDecodeOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::experimental::nlp_concat_heads_decode, device, inputSpec,
        numHeads, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(nlpConcatHeadsDecodeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<RepeatInterleaveOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, const unsigned int repeats, const int dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto repeatInterleaveOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::repeat_interleave, device, inputSpec, repeats, dim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     repeatInterleaveOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RepeatInterleaveOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const unsigned int repeats, const int dim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto repeatInterleaveOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::repeat_interleave, device, inputSpec, repeats, dim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(repeatInterleaveOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<RepeatOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> repeats,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Convert repeats to ttnn::Shape
  ::ttnn::Shape repeatShape = conversion::getShape(repeats);

  // Create query closure
  auto repeatOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(::ttnn::repeat, device,
                                               inputSpec, repeatShape);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     repeatOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RepeatOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> repeats, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Convert repeats to ttnn::Shape
  ::ttnn::Shape repeatShape = conversion::getShape(repeats);

  // Create query closure
  auto repeatOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(::ttnn::repeat, device, inputSpec,
                                           repeatShape);
  };

  return operation::getOpRuntime(repeatOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
/**
 * @brief Converts padding array to PadSpecDim format for TTNN operations.
 *
 * @param padding Array of padding values in [before0, after0, before1, after1,
 * ...] format
 * @return SmallVector of PadSpecDim objects
 */
static ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim>
convertPadding(llvm::ArrayRef<int32_t> padding) {
  ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> paddingSpec;
  // Reserve space to avoid memory reallocations
  paddingSpec.reserve((padding.size() + 1) / 2);

  constexpr int32_t defaultPadValue = 0;
  for (size_t i = 0; i < padding.size(); i += 2) {
    int32_t before = padding[i];
    int32_t after = (i + 1 < padding.size()) ? padding[i + 1] : defaultPadValue;

    assert(before >= 0 && after >= 0 && "Padding values must be non-negative");

    paddingSpec.emplace_back(static_cast<uint32_t>(before),
                             static_cast<uint32_t>(after));
  }
  return paddingSpec;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PadOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int32_t> padding,
    llvm::APFloat padValue, bool multicore, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Convert padding to PadSpecDim format
  auto paddingSpec = convertPadding(padding);

  // Create query closure
  auto padOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::pad, device, inputSpec, paddingSpec, padValue.convertToFloat(),
        multicore, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     padOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PadOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int32_t> padding, llvm::APFloat padValue, bool multicore,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Convert padding to PadSpecDim format
  auto paddingSpec = convertPadding(padding);

  // Create query closure
  auto padOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::pad, device, inputSpec, paddingSpec, padValue.convertToFloat(),
        multicore, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(padOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<SortOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, int dim, bool descending, bool stable,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto sortOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::sort, device, inputSpec, dim, descending, stable,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     sortOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SortOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int dim,
    bool descending, bool stable, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto sortOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::sort, device, inputSpec, dim, descending, stable,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(sortOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ArgMaxOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, std::optional<int32_t> dim, bool keepDim,
    bool multicore, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto argMaxOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::argmax, device, inputSpec, dim, keepDim, std::nullopt,
        multicore, detail::getNullableMemoryConfig(outputLayout), std::nullopt);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     argMaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<ArgMaxOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                TTNNLayoutAttr inputLayout,
                                std::optional<int32_t> dim, bool keepDim,
                                bool multicore, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto argMaxOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::argmax, device, inputSpec, dim, keepDim, std::nullopt,
        multicore, detail::getNullableMemoryConfig(outputLayout), std::nullopt);
  };

  return operation::getOpRuntime(argMaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ProdOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ProdOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, std::optional<int64_t> dim, bool keepDim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto prodOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::prod, device, inputSpec, dim, keepDim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     prodOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Quantization Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> QuantizationOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> scaleShape,
    TTNNLayoutAttr scaleLayout, llvm::ArrayRef<int64_t> zeroPointShape,
    TTNNLayoutAttr zeroPointLayout, std::optional<int32_t> axis,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto scaleSpecExp =
      detail::convertToTensorSpec(device, scaleShape, scaleLayout);
  if (!scaleSpecExp) {
    return scaleSpecExp.takeError();
  }
  ::ttnn::TensorSpec scaleSpec = scaleSpecExp.get();

  auto zeroPointSpecExp =
      detail::convertToTensorSpec(device, zeroPointShape, zeroPointLayout);
  if (!zeroPointSpecExp) {
    return zeroPointSpecExp.takeError();
  }
  ::ttnn::TensorSpec zeroPointSpec = zeroPointSpecExp.get();

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto quantizationOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device, inputSpec, scaleSpec,
        zeroPointSpec, axis, outputDType, outputMemoryConfig);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     quantizationOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> QuantizationOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> scaleShape, TTNNLayoutAttr scaleLayout,
    llvm::ArrayRef<int64_t> zeroPointShape, TTNNLayoutAttr zeroPointLayout,
    std::optional<int32_t> axis, std::optional<ttcore::DataType> outputDtype,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto scaleSpecExp =
      detail::convertToTensorSpec(device, scaleShape, scaleLayout);
  if (!scaleSpecExp) {
    return scaleSpecExp.takeError();
  }
  ::ttnn::TensorSpec scaleSpec = scaleSpecExp.get();

  auto zeroPointSpecExp =
      detail::convertToTensorSpec(device, zeroPointShape, zeroPointLayout);
  if (!zeroPointSpecExp) {
    return zeroPointSpecExp.takeError();
  }
  ::ttnn::TensorSpec zeroPointSpec = zeroPointSpecExp.get();

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto quantizationOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec, scaleSpec,
        zeroPointSpec, axis, outputDType, outputMemoryConfig);
  };

  return operation::getOpRuntime(quantizationOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for QuantizationOpModel.
template struct QuantizationOpModel<QuantizeOp>;
template struct QuantizationOpModel<DequantizeOp>;

//===----------------------------------------------------------------------===//
// RequantizeOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<RequantizeOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> inScaleShape,
    TTNNLayoutAttr inScaleLayout, llvm::ArrayRef<int64_t> inZeroPointShape,
    TTNNLayoutAttr inZeroPointLayout, llvm::ArrayRef<int64_t> outScaleShape,
    TTNNLayoutAttr outScaleLayout, llvm::ArrayRef<int64_t> outZeroPointShape,
    TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto inScaleSpecExp =
      detail::convertToTensorSpec(device, inScaleShape, inScaleLayout);
  if (!inScaleSpecExp) {
    return inScaleSpecExp.takeError();
  }
  ::ttnn::TensorSpec inScaleSpec = inScaleSpecExp.get();

  auto inZeroPointSpecExp =
      detail::convertToTensorSpec(device, inZeroPointShape, inZeroPointLayout);
  if (!inZeroPointSpecExp) {
    return inZeroPointSpecExp.takeError();
  }
  ::ttnn::TensorSpec inZeroPointSpec = inZeroPointSpecExp.get();

  auto outScaleSpecExp =
      detail::convertToTensorSpec(device, outScaleShape, outScaleLayout);
  if (!outScaleSpecExp) {
    return outScaleSpecExp.takeError();
  }
  ::ttnn::TensorSpec outScaleSpec = outScaleSpecExp.get();

  auto outZeroPointSpecExp = detail::convertToTensorSpec(
      device, outZeroPointShape, outZeroPointLayout);
  if (!outZeroPointSpecExp) {
    return outZeroPointSpecExp.takeError();
  }
  ::ttnn::TensorSpec outZeroPointSpec = outZeroPointSpecExp.get();

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure

  auto requantizeOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::requantize, device, inputSpec, inScaleSpec, inZeroPointSpec,
        outScaleSpec, outZeroPointSpec, axis, outputDType, outputMemoryConfig);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     requantizeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RequantizeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> inScaleShape, TTNNLayoutAttr inScaleLayout,
    llvm::ArrayRef<int64_t> inZeroPointShape, TTNNLayoutAttr inZeroPointLayout,
    llvm::ArrayRef<int64_t> outScaleShape, TTNNLayoutAttr outScaleLayout,
    llvm::ArrayRef<int64_t> outZeroPointShape,
    TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto inScaleSpecExp =
      detail::convertToTensorSpec(device, inScaleShape, inScaleLayout);
  if (!inScaleSpecExp) {
    return inScaleSpecExp.takeError();
  }
  ::ttnn::TensorSpec inScaleSpec = inScaleSpecExp.get();

  auto inZeroPointSpecExp =
      detail::convertToTensorSpec(device, inZeroPointShape, inZeroPointLayout);
  if (!inZeroPointSpecExp) {
    return inZeroPointSpecExp.takeError();
  }
  ::ttnn::TensorSpec inZeroPointSpec = inZeroPointSpecExp.get();

  auto outScaleSpecExp =
      detail::convertToTensorSpec(device, outScaleShape, outScaleLayout);
  if (!outScaleSpecExp) {
    return outScaleSpecExp.takeError();
  }
  ::ttnn::TensorSpec outScaleSpec = outScaleSpecExp.get();

  auto outZeroPointSpecExp = detail::convertToTensorSpec(
      device, outZeroPointShape, outZeroPointLayout);
  if (!outZeroPointSpecExp) {
    return outZeroPointSpecExp.takeError();
  }
  ::ttnn::TensorSpec outZeroPointSpec = outZeroPointSpecExp.get();

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto requantizeOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::requantize, device, inputSpec, inScaleSpec, inZeroPointSpec,
        outScaleSpec, outZeroPointSpec, axis, outputDType, outputMemoryConfig);
  };

  return operation::getOpRuntime(requantizeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<LinearOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
    TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto linearOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::linear, device, inputSpecA, inputSpecB, biasTensor, transposeA,
        transposeB, outputMemoryConfig, outputDType);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), deviceGrid,
                                     linearOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LinearOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto linearOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::linear, device, inputSpecA, inputSpecB, biasTensor, transposeA,
        transposeB, outputMemoryConfig, outputDType);
  };

  return operation::getOpRuntime(linearOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<MatmulOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    TTNNLayoutAttr inputLayoutA, llvm::ArrayRef<int64_t> inputShapeB,
    TTNNLayoutAttr inputLayoutB, TTNNLayoutAttr outputLayout, bool transposeA,
    bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto matmulOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::matmul, device, inputSpecA, inputSpecB, transposeA, transposeB,
        outputMemoryConfig, outputDType);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), deviceGrid,
                                     matmulOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MatmulOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecAExp =
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA);
  if (!inputSpecAExp) {
    return inputSpecAExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecA = inputSpecAExp.get();

  auto inputSpecBExp =
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB);
  if (!inputSpecBExp) {
    return inputSpecBExp.takeError();
  }
  ::ttnn::TensorSpec inputSpecB = inputSpecBExp.get();

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto matmulOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(::ttnn::matmul, device, inputSpecA,
                                           inputSpecB, transposeA, transposeB,
                                           outputMemoryConfig, outputDType);
  };

  return operation::getOpRuntime(matmulOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// DeallocateOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<DeallocateOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, bool force) {
  // sgholamiTT: DeallocateOp's invoke method in tt-metal returns void. So it
  // cannot be called via a call to query_op_constraints (see
  // extract_output_tensor usage). Besides, DeallocateOp has no memory usage as
  // it simply deallocates memory. So I decided to return an empty
  // OpConstraints, instead of returning an error.
  return OpConstraints{};
}

llvm::Expected<size_t>
OpModel<DeallocateOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                    TTNNLayoutAttr inputLayout, bool force) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto deallocateOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(::ttnn::deallocate, device,
                                           inputSpec, force);
  };

  return operation::getOpRuntime(deallocateOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<FillCacheOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> cacheShape,
    TTNNLayoutAttr cacheLayout, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, uint32_t batchOffset,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  auto cacheSpecExp =
      detail::convertToTensorSpec(device, cacheShape, cacheLayout);
  if (!cacheSpecExp) {
    return cacheSpecExp.takeError();
  }
  ::ttnn::TensorSpec cacheSpec = cacheSpecExp.get();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto fillCacheOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::fill_cache, device, cacheSpec, inputSpec, batchOffset);
  };

  return operation::getOpConstraints(cacheLayout.getContext(), deviceGrid,
                                     fillCacheOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<FillCacheOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  auto cacheSpecExp =
      detail::convertToTensorSpec(device, cacheShape, cacheLayout);
  if (!cacheSpecExp) {
    return cacheSpecExp.takeError();
  }
  ::ttnn::TensorSpec cacheSpec = cacheSpecExp.get();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto fillCacheOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(::ttnn::fill_cache, device,
                                           cacheSpec, inputSpec, batchOffset);
  };

  return operation::getOpRuntime(fillCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<UpdateCacheOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> cacheShape,
    TTNNLayoutAttr cacheLayout, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> updateIndexShape,
    TTNNLayoutAttr updateIndexLayout, uint32_t batchOffset,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  auto cacheSpecExp =
      detail::convertToTensorSpec(device, cacheShape, cacheLayout);
  if (!cacheSpecExp) {
    return cacheSpecExp.takeError();
  }
  ::ttnn::TensorSpec cacheSpec = cacheSpecExp.get();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // TODO(#1510): modify the ttnn::update_cache to take a tensor for
  // updateIndex.
  // UpdateIndex is stored as a tensor in mlir, but the ttnn::update_cache
  // expects a scalar uint32_t. So we need to extract the scalar value from the
  // tensor which is not possible in compile time (as opposed to the workaround
  // that is implemented in runtime code in PR 1437). So we use a default value
  // of 0.
  uint32_t updateIdx = 0; // Default to first position
  (void)updateIndexShape;
  (void)updateIndexLayout;

  auto updateCacheOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(::ttnn::update_cache, device,
                                               cacheSpec, inputSpec, updateIdx,
                                               batchOffset);
  };

  return operation::getOpConstraints(cacheLayout.getContext(), deviceGrid,
                                     updateCacheOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<UpdateCacheOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  auto cacheSpecExp =
      detail::convertToTensorSpec(device, cacheShape, cacheLayout);
  if (!cacheSpecExp) {
    return cacheSpecExp.takeError();
  }
  ::ttnn::TensorSpec cacheSpec = cacheSpecExp.get();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // TODO(#1510): modify the ttnn::update_cache to take a tensor for
  // updateIndex.
  uint32_t updateIdx = 0; // Default to first position
  (void)updateIndexShape;
  (void)updateIndexLayout;

  auto updateCacheOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(::ttnn::update_cache, device,
                                           cacheSpec, inputSpec, updateIdx,
                                           batchOffset);
  };

  return operation::getOpRuntime(updateCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<Conv2dOp>::getOpConstraints(
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
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig,
          biasLayout.has_value(), /*transpose*/ false);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  // Prepare bias tensor if present.
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            input_height, input_width, kernel_size, stride, padding, dilation,
            groups, conv2dConfig);
    if (!preparedBiasExp) {
      return preparedBiasExp.takeError();
    }
    biasSpec = preparedBiasExp.get();
  }

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasSpec, conv2dConfigConverted,
        deviceComputeKernelConfigConverted,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     conv2dOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<Conv2dOp>::getOpRuntime(
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
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig,
          biasLayout.has_value(), /*transpose*/ false);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }

  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  // Prepare bias tensor if present.
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            input_height, input_width, kernel_size, stride, padding, dilation,
            groups, conv2dConfig);
    if (!preparedBiasExp) {
      return preparedBiasExp.takeError();
    }
    biasSpec = preparedBiasExp.get();
  }

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  auto conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);
  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);
  // Create query closure
  auto conv2dOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasSpec, conv2dConfigConverted,
        deviceComputeKernelConfigConverted,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(conv2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ConvTranspose2dOp>::getOpConstraints(
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
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig,
          biasLayout.has_value(), /*transpose*/ true);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    // TODO(odjuricic): This might be really slow. Needs to be done within graph
    // capture block.
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto convTranspose2dOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::conv_transpose2d, device, inputSpec, weightSpec, device,
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(output_padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasTensor, conv2dConfigConverted,
        /* compute_config */ std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     convTranspose2dOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ConvTranspose2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> output_padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig,
          biasLayout.has_value(), /*transpose*/ true);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }

  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto convTranspose2dOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::conv_transpose2d, device, inputSpec, weightSpec, device,
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(output_padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasTensor, conv2dConfigConverted,
        /* compute_config */ std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(convTranspose2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConv2dWeightsOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<PrepareConv2dWeightsOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> weightShape, MemoryConfigAttr inputMemConfig,
    ::mlir::tt::ttnn::Layout inputTensorLayout, llvm::StringRef weightsFormat,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool hasBias, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(weightLayout != nullptr && "Weight layout is nullptr");

  if (weightLayout.getBufferType() != BufferType::SystemMemory) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d weight tensor assumed to be on host.");
  }
  if (weightLayout.getDataType() != ttcore::DataType::Float32 &&
      weightLayout.getDataType() != ttcore::DataType::BFloat16) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d weight tensor assumed to be float32 or bfloat16.");
  }
  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());
  // Read output data type from output layout (if present) or from outputDtype.
  std::optional<::tt::tt_metal::DataType> convertedOutputDtype = std::nullopt;
  if (outputLayout) {
    convertedOutputDtype = conversion::getDataType(outputLayout.getDataType());
  } else if (outputDtype.has_value()) {
    convertedOutputDtype = conversion::getDataType(outputDtype.value());
  }
  // The following parameter does not exist in the op yet.
  std::optional<::ttnn::operations::conv::conv2d::Conv2dSliceConfig>
      sliceConfigConverted = std::nullopt;

  auto prepareConv2dWeightsQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv2d::prepare_conv_weights, device,
        weightTensor, conversion::getMemoryConfig(inputMemConfig),
        conversion::getPageLayout(inputTensorLayout), weightsFormat.str(),
        inChannels, outChannels, batchSize, inputHeight, inputWidth,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, conversion::getDataType(inputDtype),
        convertedOutputDtype, conversion::getConv2dConfig(conv2dConfig),
        conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig),
        sliceConfigConverted);
  };

  return operation::getOpConstraints(weightLayout.getContext(), deviceGrid,
                                     prepareConv2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<PrepareConv2dBiasOp>::getOpConstraints(
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
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(biasLayout != nullptr && "Weight layout is nullptr");

  if (biasLayout.getBufferType() != BufferType::SystemMemory) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Conv2d bias tensor assumed to be on host.");
  }
  if (biasLayout.getDataType() != ttcore::DataType::Float32 &&
      biasLayout.getDataType() != ttcore::DataType::BFloat16) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d bias tensor assumed to be float32 or bfloat16.");
  }
  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());
  // Read output data type from output layout (if present) or from outputDtype.
  std::optional<::tt::tt_metal::DataType> convertedOutputDtype = std::nullopt;
  if (outputLayout) {
    convertedOutputDtype = conversion::getDataType(outputLayout.getDataType());
  } else if (outputDtype.has_value()) {
    convertedOutputDtype = conversion::getDataType(outputDtype.value());
  }

  auto prepareConv2dWeightsQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv2d::prepare_conv_bias, device,
        biasTensor, conversion::getMemoryConfig(inputMemConfig),
        conversion::getPageLayout(inputTensorLayout), inChannels, outChannels,
        batchSize, inputHeight, inputWidth,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, device, conversion::getDataType(inputDtype),
        convertedOutputDtype, conversion::getConv2dConfig(conv2dConfig),
        conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig));
  };

  return operation::getOpConstraints(biasLayout.getContext(), deviceGrid,
                                     prepareConv2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MaxPool2D
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<MaxPool2dOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, int32_t batchSize, int32_t inputHeight,
    int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, bool inPlaceHalo, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto maxPool2DQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */, inPlaceHalo,
        false /* return_indices */);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     maxPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MaxPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool inPlaceHalo,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto maxPool2DQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */, inPlaceHalo);
  };

  return operation::getOpRuntime(maxPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AvgPool2D
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<AvgPool2dOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, int32_t batchSize, int32_t inputHeight,
    int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, bool inPlaceHalo, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // default values for the variables that are received by the op's invoke
  // method in tt-metal but do not exist in the op's definition in TTNNOps.td:
  bool countIncludePad = true;
  std::optional<int32_t> divisorOverride = std::nullopt;

  // Create query closure
  auto avgPool2DQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::avg_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        ceilMode, countIncludePad, divisorOverride,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */, inPlaceHalo);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     avgPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<AvgPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool inPlaceHalo,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // default values for the variables that are received by the op's invoke
  // method in tt-metal but do not exist in the op's definition in TTNNOps.td:
  bool countIncludePad = true;
  std::optional<int32_t> divisorOverride = std::nullopt;

  // Create query closure
  auto avgPool2DQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::avg_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToMultiSizeStdArray<uint32_t, 2, 4>(
            padding),
        ceilMode, countIncludePad, divisorOverride,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */, inPlaceHalo);
  };

  return operation::getOpRuntime(avgPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// GlobalAvgPool2dOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<GlobalAvgPool2dOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto globalAvgPool2DQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::global_avg_pool2d, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     globalAvgPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<GlobalAvgPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto globalAvgPool2DQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::global_avg_pool2d, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(globalAvgPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// BatchNormOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<BatchNormOp>::getOpConstraints(
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
    bool training, llvm::APFloat momentum, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::ttnn::TensorSpec> runningMeanSpec =
      detail::convertToOptionalTensorSpec(device, runningMeanShape,
                                          runningMeanLayout);
  std::optional<::ttnn::TensorSpec> runningVarSpec =
      detail::convertToOptionalTensorSpec(device, runningVarShape,
                                          runningVarLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);
  // The following arguments are received by the invoke method of batch norm but
  // they don't exist in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  auto batchNormQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::batch_norm, device, inputSpec, runningMeanSpec, runningVarSpec,
        training, epsilon.convertToFloat(), momentum.convertToFloat(),
        weightSpec, biasSpec, outputSpec,
        detail::getNullableMemoryConfig(outputLayout), computeKernelConfig);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     batchNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<BatchNormOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    bool training, llvm::APFloat momentum, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::ttnn::TensorSpec> runningMeanSpec =
      detail::convertToOptionalTensorSpec(device, runningMeanShape,
                                          runningMeanLayout);
  std::optional<::ttnn::TensorSpec> runningVarSpec =
      detail::convertToOptionalTensorSpec(device, runningVarShape,
                                          runningVarLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);
  // The following arguments are received by the invoke method of batch norm but
  // they don't exist in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;
  // Create query closure
  auto batchNormQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::batch_norm, device, inputSpec, runningMeanSpec, runningVarSpec,
        training, epsilon.convertToFloat(), momentum.convertToFloat(),
        weightSpec, biasSpec, outputSpec,
        detail::getNullableMemoryConfig(outputLayout), computeKernelConfig);
  };

  return operation::getOpRuntime(batchNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<RMSNormOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  // This information is not available in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> residualInputSpec = std::nullopt;

  // Create query closure
  auto rmsNormQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::rms_norm, device, inputSpec, epsilon.convertToFloat(),
        weightSpec, biasSpec, residualInputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     rmsNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RMSNormOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  // This information is not available in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> residualInputSpec = std::nullopt;

  // Create query closure
  auto rmsNormQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::rms_norm, device, inputSpec, epsilon.convertToFloat(),
        weightSpec, biasSpec, residualInputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(rmsNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampScalar
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ClampScalarOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::APFloat min, llvm::APFloat max,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert float
  float minVal = min.convertToFloat();
  float maxVal = max.convertToFloat();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto clampScalarQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::clamp, device, inputSpec, minVal, maxVal,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     clampScalarQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ClampScalarOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat min, llvm::APFloat max, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert float
  float minVal = min.convertToFloat();
  float maxVal = max.convertToFloat();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto clampScalarQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::clamp, device, inputSpec, minVal, maxVal,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(clampScalarQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampTensor
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ClampTensorOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> minShape,
    TTNNLayoutAttr minLayout, llvm::ArrayRef<int64_t> maxShape,
    TTNNLayoutAttr maxLayout, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto minSpecExp = detail::convertToTensorSpec(device, minShape, minLayout);
  if (!minSpecExp) {
    return minSpecExp.takeError();
  }
  ::ttnn::TensorSpec minSpec = minSpecExp.get();

  auto maxSpecExp = detail::convertToTensorSpec(device, maxShape, maxLayout);
  if (!maxSpecExp) {
    return maxSpecExp.takeError();
  }
  ::ttnn::TensorSpec maxSpec = maxSpecExp.get();

  // Create query closure
  auto clampTensorQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::clamp, device, inputSpec, minSpec, maxSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     clampTensorQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ClampTensorOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> minShape, TTNNLayoutAttr minLayout,
    llvm::ArrayRef<int64_t> maxShape, TTNNLayoutAttr maxLayout,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto minSpecExp = detail::convertToTensorSpec(device, minShape, minLayout);
  if (!minSpecExp) {
    return minSpecExp.takeError();
  }
  ::ttnn::TensorSpec minSpec = minSpecExp.get();

  auto maxSpecExp = detail::convertToTensorSpec(device, maxShape, maxLayout);
  if (!maxSpecExp) {
    return maxSpecExp.takeError();
  }
  ::ttnn::TensorSpec maxSpec = maxSpecExp.get();

  // Create query closure
  auto clampTensorQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::clamp, device, inputSpec, minSpec, maxSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(clampTensorQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Permute
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<PermuteOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> permutation,
    llvm::APFloat padValue, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert permutations of TTNN_PermuteOp to dims of ttnn::permute
  ::ttsl::SmallVector<int64_t> dims(permutation.size());
  std::copy(permutation.begin(), permutation.end(), dims.begin());

  float defaultedPadValue = padValue.convertToFloat();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto permuteQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::permute, device, inputSpec, dims,
        detail::getNullableMemoryConfig(outputLayout), defaultedPadValue);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     permuteQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PermuteOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert permutations of TTNN_PermuteOp to dims of ttnn::permute
  ::ttsl::SmallVector<int64_t> dims(permutation.size());
  std::copy(permutation.begin(), permutation.end(), dims.begin());

  // Convert float
  float defaultedPadValue = padValue.convertToFloat();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto permuteQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::permute, device, inputSpec, dims,
        detail::getNullableMemoryConfig(outputLayout), defaultedPadValue);
  };

  return operation::getOpRuntime(permuteQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Upsample
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<UpsampleOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, mlir::Attribute scaleFactor,
    llvm::StringRef mode, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert params
  std::variant<int, ::tt::tt_metal::Array2D> convertedScaleFactor;
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(scaleFactor)) {
    convertedScaleFactor = static_cast<int>(value.getSInt());
  } else if (auto tuple =
                 mlir::dyn_cast<::mlir::detail::DenseArrayAttrImpl<int32_t>>(
                     scaleFactor);
             tuple.size() == 2) {
    std::array<uint32_t, 2> arr;
    arr[0] = static_cast<uint32_t>(tuple[0]);
    arr[1] = static_cast<uint32_t>(tuple[1]);
    convertedScaleFactor = ::tt::tt_metal::Array2D(arr);
  } else {
    return llvm::createStringError("Invalid scaleFactor");
  }

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto upsampleQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::upsample, device, inputSpec, convertedScaleFactor,
        std::string(mode), detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     upsampleQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<UpsampleOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute scaleFactor, llvm::StringRef mode,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert parameters
  std::variant<int, ::tt::tt_metal::Array2D> convertedScaleFactor;
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(scaleFactor)) {
    convertedScaleFactor = static_cast<int>(value.getSInt());
  } else if (auto tuple =
                 mlir::dyn_cast<::mlir::detail::DenseArrayAttrImpl<int32_t>>(
                     scaleFactor);
             tuple.size() == 2) {
    std::array<uint32_t, 2> arr;
    arr[0] = static_cast<uint32_t>(tuple[0]);
    arr[1] = static_cast<uint32_t>(tuple[1]);
    convertedScaleFactor = ::tt::tt_metal::Array2D(arr);
  } else {
    return llvm::createStringError("Invalid scaleFactor");
  }

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  // Create query closure
  auto upsampleQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::upsample, device, inputSpec, convertedScaleFactor,
        std::string(mode), detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(upsampleQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
struct EmbeddingOpArgs {
  ::ttnn::TensorSpec inputSpec;
  ::ttnn::TensorSpec weightSpec;
  std::optional<::ttnn::TensorSpec> outputSpec;
};

llvm::Expected<EmbeddingOpArgs>
getEmbeddingOpArgs(::tt::tt_metal::distributed::MeshDevice *device,
                   llvm::ArrayRef<int64_t> inputShape,
                   TTNNLayoutAttr inputLayout,
                   llvm::ArrayRef<int64_t> weightShape,
                   TTNNLayoutAttr weightLayout, TTNNLayoutAttr outputLayout) {
  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto weightSpecExp =
      detail::convertToTensorSpec(device, weightShape, weightLayout);
  if (!weightSpecExp) {
    return weightSpecExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = weightSpecExp.get();

  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  if (outputLayout) {
    auto outputSpecExp =
        detail::convertToTensorSpec(device, weightShape, outputLayout);
    if (!outputSpecExp) {
      return outputSpecExp.takeError();
    }
    outputSpec = outputSpecExp.get();
  }

  return EmbeddingOpArgs{inputSpec, weightSpec, outputSpec};
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<EmbeddingOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> weightShape,
    TTNNLayoutAttr weightLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  llvm::Expected<EmbeddingOpArgs> embeddingOpArgsExp = getEmbeddingOpArgs(
      device, inputShape, inputLayout, weightShape, weightLayout, outputLayout);
  if (!embeddingOpArgsExp) {
    return embeddingOpArgsExp.takeError();
  }
  EmbeddingOpArgs &embeddingOpArgs = embeddingOpArgsExp.get();

  // sgholamiTT: For the following arguments, I tried to follow the same pattern
  // as in the runtime/embedding.cpp. Subject to change in the future.
  std::optional<int> padToken = std::nullopt;
  ::ttnn::Layout layout =
      outputLayout ? (outputLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT)
                   : (weightLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT);
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  std::optional<::ttnn::DataType> dtype =
      outputLayout ? std::make_optional(
                         conversion::getDataType(outputLayout.getDataType()))
                   : std::nullopt;

  auto embeddingOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::embedding, device, embeddingOpArgs.inputSpec,
        embeddingOpArgs.weightSpec, padToken, layout, embeddingsType, dtype,
        detail::getNullableMemoryConfig(outputLayout),
        embeddingOpArgs.outputSpec);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     embeddingOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<EmbeddingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  llvm::Expected<EmbeddingOpArgs> embeddingOpArgsExp = getEmbeddingOpArgs(
      device, inputShape, inputLayout, weightShape, weightLayout, outputLayout);
  if (!embeddingOpArgsExp) {
    return embeddingOpArgsExp.takeError();
  }
  EmbeddingOpArgs &embeddingOpArgs = embeddingOpArgsExp.get();

  // sgholamiTT: For the following arguments, I tried to follow the same pattern
  // as in the runtime/embedding.cpp. Subject to change in the future.
  std::optional<int> padToken = std::nullopt;
  ::ttnn::Layout layout =
      outputLayout ? (outputLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT)
                   : (weightLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT);
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  std::optional<::ttnn::DataType> dtype =
      outputLayout ? std::make_optional(
                         conversion::getDataType(outputLayout.getDataType()))
                   : std::nullopt;

  auto embeddingOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::embedding, device, embeddingOpArgs.inputSpec,
        embeddingOpArgs.weightSpec, padToken, layout, embeddingsType, dtype,
        detail::getNullableMemoryConfig(outputLayout),
        embeddingOpArgs.outputSpec);
  };

  return operation::getOpRuntime(embeddingOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<EmbeddingBackwardOp>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> weightShape,
    TTNNLayoutAttr weightLayout, llvm::ArrayRef<int64_t> inGradientShape,
    TTNNLayoutAttr inGradientLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto weightSpecExp =
      detail::convertToTensorSpec(device, weightShape, weightLayout);
  if (!weightSpecExp) {
    return weightSpecExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = weightSpecExp.get();

  auto inGradientSpecExp =
      detail::convertToTensorSpec(device, inGradientShape, inGradientLayout);
  if (!inGradientSpecExp) {
    return inGradientSpecExp.takeError();
  }
  ::ttnn::TensorSpec inGradientSpec = inGradientSpecExp.get();

  auto embeddingBackwardOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::embedding_bw, device, inputSpec, weightSpec, inGradientSpec,
        /*dtype*/ std::nullopt, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     embeddingBackwardOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<mlir::tt::ttnn::EmbeddingBackwardOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> inGradientShape, TTNNLayoutAttr inGradientLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  auto weightSpecExp =
      detail::convertToTensorSpec(device, weightShape, weightLayout);
  if (!weightSpecExp) {
    return weightSpecExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = weightSpecExp.get();

  auto inGradientSpecExp =
      detail::convertToTensorSpec(device, inGradientShape, inGradientLayout);
  if (!inGradientSpecExp) {
    return inGradientSpecExp.takeError();
  }
  ::ttnn::TensorSpec inGradientSpec = inGradientSpecExp.get();

  auto embeddingBackwardOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::embedding_bw, device, inputSpec, weightSpec, inGradientSpec,
        /*dtype*/ std::nullopt, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(embeddingBackwardOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::EmptyOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttcore::DataTypeAttr dtype, mlir::tt::ttnn::Layout inputLayout,
    mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Use the output layout if possible:
  ::tt::tt_metal::MemoryConfig memConfig =
      outputLayout ? conversion::getMemoryConfig(outputLayout)
                   : conversion::getMemoryConfig(memoryConfig);

  auto emptyOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::empty, device, conversion::getShape(inputShape),
        conversion::getDataType(dtype.getValue()),
        conversion::getPageLayout(inputLayout), device, memConfig);
  };

  return operation::getOpConstraints(dtype.getContext(), deviceGrid,
                                     emptyOpQuery);
#else
  return OpConstraints{};
#endif //
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//
// sgholamiTT: There are two reasons why receiving the start, end, and step as
// attributes is better than as integers:
//   1. That is the only valid way to aquire a pointer to MLIRContext.
//   2. Using getInt() member function of ::mlir::IntegerAttr is safer and more
//      mlir idiomatic than static_cast<int64_t>(start).
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ArangeOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, ::mlir::IntegerAttr start,
    ::mlir::IntegerAttr end, ::mlir::IntegerAttr step,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    std::optional<mlir::tt::ttnn::MemoryConfigAttr> memConfig,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  // ~~~~~~~~~~~~~~~~~~~~~ Note ~~~~~~~~~~~~~~~~~~~~~
  // The following default values are taken from Arrange's invoke function in
  // tt-metal/ttnn/cpp/ttnn/operations/creation.hpp
  const ::tt::tt_metal::DataType defaultDtypeInMetal =
      ::tt::tt_metal::DataType::BFLOAT16;
  const ::ttnn::MemoryConfig defaultMemoryConfigInMetal =
      ::ttnn::DRAM_MEMORY_CONFIG;
  const ::ttnn::Layout defaultLayoutInMetal = ::ttnn::ROW_MAJOR_LAYOUT;
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ::tt::tt_metal::DataType dataType = defaultDtypeInMetal;
  if (dtype.has_value()) {
    dataType = conversion::getDataType(dtype.value());
  }
  ::ttnn::MemoryConfig memoryConfig = defaultMemoryConfigInMetal;
  // Prefer the output layout if possible:
  if (outputLayout) {
    memoryConfig = conversion::getMemoryConfig(outputLayout);
  } else if (memConfig.has_value()) {
    memoryConfig = conversion::getMemoryConfig(memConfig.value());
  }
  std::optional<std::reference_wrapper<::tt::tt_metal::distributed::MeshDevice>>
      deviceRef = *device;

  auto arangeOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::arange, device, start.getInt(), end.getInt(), step.getInt(),
        dataType, deviceRef, memoryConfig, defaultLayoutInMetal);
  };

  return operation::getOpConstraints(start.getContext(), deviceGrid,
                                     arangeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<mlir::tt::ttnn::FullOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, mlir::tt::ttnn::ShapeAttr shape,
    mlir::Attribute fillValue, std::optional<mlir::tt::ttcore::DataType> dtype,
    std::optional<mlir::tt::ttnn::Layout> layout,
    std::optional<mlir::tt::ttnn::MemoryConfigAttr> memoryConfig,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prefer the output layout if possible:
  std::optional<::ttnn::MemoryConfig> metalMemConfig = std::nullopt;
  if (outputLayout) {
    metalMemConfig = conversion::getMemoryConfig(outputLayout);
  } else if (memoryConfig.has_value()) {
    metalMemConfig = conversion::getMemoryConfig(memoryConfig.value());
  }

  std::optional<::ttnn::DataType> metalDtype = std::nullopt;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }
  ::ttnn::Shape metalShape = conversion::getShape(shape.getShape());

  std::optional<::ttnn::Layout> metalLayout = std::nullopt;
  if (layout.has_value()) {
    metalLayout = conversion::getPageLayout(layout.value());
  }
  std::optional<std::reference_wrapper<::tt::tt_metal::distributed::MeshDevice>>
      deviceRef = *device;

  // Helper lambda to create the query with any fill value type
  auto createFullOpQuery = [=](auto convertedFillValue) {
    return [=]() {
      return ::ttnn::graph::query_op_constraints(
          ::ttnn::full, device, metalShape, convertedFillValue, metalDtype,
          metalLayout, deviceRef, metalMemConfig,
          /*optional_output_tensor = */ std::nullopt);
    };
  };

  // The invoke function of fullOp is templated over the fill value type. That's
  // why the following code is aranged in this way.
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(fillValue)) {
    int convertedFillValue = static_cast<int>(value.getInt());
    auto query = createFullOpQuery(convertedFillValue);
    return operation::getOpConstraints(fillValue.getContext(), deviceGrid,
                                       query);
  }
  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(fillValue)) {
    float convertedFillValue = value.getValue().convertToFloat();
    auto query = createFullOpQuery(convertedFillValue);
    return operation::getOpConstraints(fillValue.getContext(), deviceGrid,
                                       query);
  }
  return llvm::createStringError("Invalid fillValue");
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RandOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<mlir::tt::ttnn::RandOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, mlir::tt::ttnn::ShapeAttr size,
    mlir::tt::ttcore::DataType dtype,
    mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
    mlir::tt::ttnn::Layout layout, llvm::APFloat low, llvm::APFloat high,
    uint32_t seed, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prefer the output layout if possible:
  ::ttnn::MemoryConfig metalMemConfig = ::ttnn::DRAM_MEMORY_CONFIG;
  if (outputLayout) {
    metalMemConfig = conversion::getMemoryConfig(outputLayout);
  } else if (memoryConfig) {
    metalMemConfig = conversion::getMemoryConfig(memoryConfig);
  }

  auto randOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::rand, device, conversion::getShape(size.getShape()),
        std::ref(*device), conversion::getDataType(dtype),
        conversion::getPageLayout(layout), metalMemConfig, low.convertToFloat(),
        high.convertToFloat(), seed);
  };

  return operation::getOpConstraints(size.getContext(), deviceGrid,
                                     randOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
// sgholamiTT: I decided to not promote these helper methods to conversion.hpp
// for two reasons:
//   1. There's no other clear usage for them.
//   2. Some of them are specialized for ConstantOp.

mlir::Type getElementType(mlir::ElementsAttr value) {
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(value)) {
    return denseAttr.getType().getElementType();
  }
  if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(value)) {
    return splatAttr.getType().getElementType();
  }
  assert(false && "Unknown constant value attribute type");
}

::ttnn::Shape getShape(mlir::ElementsAttr value) {
  if (auto rankedTensorType =
          dyn_cast<mlir::RankedTensorType>(value.getType())) {
    // Get the shape as a vector of dimensions
    llvm::ArrayRef<int64_t> shape = rankedTensorType.getShape();
    return conversion::getShape(shape);
  }
  assert(false && "Unknown constant value attribute type");
}

::tt::tt_metal::DataType getDataType(const mlir::ElementsAttr attr) {
  ::mlir::Type elType = getElementType(attr);
  ::tt::tt_metal::DataType dtype = ::tt::tt_metal::DataType::INVALID;
  if (elType.isBF16()) {
    dtype = ::tt::tt_metal::DataType::BFLOAT16;
  } else if (elType.isF32()) {
    dtype = ::tt::tt_metal::DataType::FLOAT32;
  } else if (elType.isUnsignedInteger(32)) {
    dtype = ::tt::tt_metal::DataType::UINT32;
  } else if (elType.isUnsignedInteger(16)) {
    dtype = ::tt::tt_metal::DataType::UINT16;
  } else if (elType.isUnsignedInteger(8)) {
    dtype = ::tt::tt_metal::DataType::UINT8;
  } else if (elType.isInteger(32)) {
    dtype = ::tt::tt_metal::DataType::INT32;
  }
  assert(dtype != ::tt::tt_metal::DataType::INVALID && "Unsupported data type");
  return dtype;
}

// Helper macro to reduce repetition in type dispatch
#define DISPATCH_TYPE(TYPE_CHECK, CPP_TYPE)                                    \
  if (elType.TYPE_CHECK) {                                                     \
    auto rawDataExp = detail::getRawDataFromElementsAttr<CPP_TYPE>(value);     \
    if (!rawDataExp) {                                                         \
      return rawDataExp.takeError();                                           \
    }                                                                          \
    return func(rawDataExp.get());                                             \
  }

// Helper function to dispatch getRawDataFromElementsAttr based on element type
// (we use this technique since from_buffer op in metal is templated over the
// input vector type.)
template <typename Func>
auto dispatchGetRawData(mlir::ElementsAttr value, Func &&func)
    -> decltype(func(std::declval<std::vector<int32_t>>())) {
  // from_span<T> has template instantiations for the following types:
  // int32_t, uint8_t, uint16_t, uint32_t, bfloat16.
  // We support all of these types:
  ::mlir::Type elType = getElementType(value);
  DISPATCH_TYPE(isUnsignedInteger(8), uint8_t)
  DISPATCH_TYPE(isUnsignedInteger(16), uint16_t)
  DISPATCH_TYPE(isUnsignedInteger(32), uint32_t)
  DISPATCH_TYPE(isInteger(32), int32_t)
  DISPATCH_TYPE(isF32(), float)
  DISPATCH_TYPE(isBF16(), bfloat16)

  return llvm::createStringError(std::errc::invalid_argument,
                                 "Unsupported element type for ConstantOp");
}

#undef DISPATCH_TYPE
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<ConstantOp>::getOpConstraints(ttcore::GridAttr deviceGrid,
                                      mlir::ElementsAttr value,
                                      TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<::tt::tt_metal::Layout> metalLayout = std::nullopt;
  if (outputLayout) {
    metalLayout = conversion::getPageLayout(outputLayout);
  }
  auto func = [&](auto rawData) {
    auto constantOpQuery = [=]() {
      return ::ttnn::graph::query_op_constraints(
          ::ttnn::from_buffer, device, rawData, getShape(value),
          getDataType(value), device, metalLayout,
          detail::getNullableMemoryConfig(outputLayout));
    };
    return operation::getOpConstraints(value.getContext(), deviceGrid,
                                       constantOpQuery);
  };
  return dispatchGetRawData(value, func);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::ttnn::op_model
