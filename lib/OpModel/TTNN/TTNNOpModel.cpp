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
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>

#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::op_model::ttnn {

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
               mlir::tt::ttcore::GridAttr workerGrid) {
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
                    ::llvm::ArrayRef<int64_t> shape,
                    ::mlir::tt::ttnn::TTNNLayoutAttr layout) {
  const ::ttnn::TensorSpec spec = conversion::getTensorSpec(shape, layout);
  if (conversion::validateTensorSpec(
          spec, device->compute_with_storage_grid_size())) {
    return spec;
  }

  return llvm::createStringError(
      "Unable to create TensorSpec out of given shape and layout");
}

/**
 * @brief Convenience wrapper to get a memory config from a TTNNLayout attr that
 * may be a nullptr. Returns std::nullopt if layout is nullptr
 */
std::optional<::tt::tt_metal::MemoryConfig>
getNullableMemoryConfig(::mlir::tt::ttnn::TTNNLayoutAttr layout) {
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
getNullableDataType(::mlir::tt::ttnn::TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getDataType(layout.getDataType());
}

template <typename OpTy>
auto getOpSymbol() {
  if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ReluOp>) {
    return ::ttnn::relu;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::SqrtOp>) {
    return ::ttnn::sqrt;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::SinOp>) {
    return ::ttnn::sin;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::CosOp>) {
    return ::ttnn::cos;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::TanhOp>) {
    return ::ttnn::tanh;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LogOp>) {
    return ::ttnn::log;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ReciprocalOp>) {
    return ::ttnn::reciprocal;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::AddOp>) {
    return ::ttnn::add;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::MultiplyOp>) {
    return ::ttnn::multiply;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::SubtractOp>) {
    return ::ttnn::subtract;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::DivideOp>) {
    return ::ttnn::divide;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::EqualOp>) {
    return ::ttnn::eq;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::NotEqualOp>) {
    return ::ttnn::ne;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::GreaterEqualOp>) {
    return ::ttnn::ge;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::GreaterThanOp>) {
    return ::ttnn::gt;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LessEqualOp>) {
    return ::ttnn::le;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LessThanOp>) {
    return ::ttnn::lt;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LogicalAndOp>) {
    return ::ttnn::logical_and;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LogicalOrOp>) {
    return ::ttnn::logical_or;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::LogicalXorOp>) {
    return ::ttnn::logical_xor;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::MaximumOp>) {
    return ::ttnn::maximum;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::MinimumOp>) {
    return ::ttnn::minimum;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::WhereOp>) {
    return ::ttnn::where;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::MeanOp>) {
    return ::ttnn::mean;
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::SumOp>) {
    return ::ttnn::sum;
  } else {
    static_assert(ttmlir::utils::always_false(),
                  "add mapping from TTNN dialect to TTNN lib op");
  }
}

} // namespace detail
#endif // TTMLIR_ENABLE_OPMODEL

bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr layout,
                                 mlir::tt::ttcore::GridAttr maxGrid) {
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
                      ::mlir::tt::ttcore::DataType dataType) {
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig, bool hasBias,
    bool transpose) {
  if (weightLayout.getBufferType() !=
      mlir::tt::ttnn::BufferType::SystemMemory) {
    llvm::report_fatal_error("Conv2d weight tensor assumed to be on host.");
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
        &::ttnn::operations::conv::conv2d::prepare_conv_weights<
            ::tt::tt_metal::distributed::MeshDevice>,
        device, weightTensor, inputSpec.memory_config(), inputSpec.layout(),
        "OIHW", in_channels, out_channels, batch_size, input_height,
        input_width,
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
            prepare_conv_transpose2d_weights<
                ::tt::tt_metal::distributed::MeshDevice>,
        device, weightTensor, inputSpec.memory_config(), inputSpec.layout(),
        "IOHW", in_channels, out_channels, batch_size, input_height,
        input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> biasShape,
    mlir::tt::ttnn::TTNNLayoutAttr biasLayout,
    ::tt::tt_metal::DataType weightsDtype, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig) {
  if (biasLayout.getBufferType() != mlir::tt::ttnn::BufferType::SystemMemory) {
    llvm::report_fatal_error("Conv2d bias tensor assumed to be on host.");
  }

  // TODO(rpavlovicTT):: Move this to tt-metal side #4043
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

  auto prepare_fn = &::ttnn::operations::conv::conv2d::prepare_conv_bias<
      ::tt::tt_metal::distributed::MeshDevice>;
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
getPreparedConv2dWeightsOutputTensor(mlir::tt::ttnn::Conv2dOp *op) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto input = op->getInput().getType();
  auto weight = op->getWeight().getType();
  auto inputLayout =
      mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(input.getEncoding());
  auto weightLayout =
      mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(weight.getEncoding());

  llvm::Expected<::ttnn::TensorSpec> outputTensorSpec =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          input.getShape(), inputLayout, weight.getShape(), weightLayout,
          op->getInChannels(), op->getOutChannels(), op->getBatchSize(),
          op->getInputHeight(), op->getInputWidth(), op->getKernelSize(),
          op->getStride(), op->getPadding(), op->getDilation(), op->getGroups(),
          op->getConv2dConfig(), op->getBias() != nullptr,
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

llvm::Expected<bool>
Device::getDeviceConstraints(mlir::tt::ttcore::GridAttr workerGrid) {
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
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<size_t> UnaryEltwiseOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

// Explicit template instantiation for UnaryEltwiseOpModel.
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::ReluOp>;
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::SqrtOp>;
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::SinOp>;
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::CosOp>;
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::TanhOp>;
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::LogOp>;
template struct UnaryEltwiseOpModel<mlir::tt::ttnn::ReciprocalOp>;

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::SigmoidOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::SigmoidOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
// ExpOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<mlir::tt::ttnn::ExpOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
  bool fastApproxMode = false;

  // Create query closure
  auto expOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::exp, device, inputSpec, fastApproxMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     expOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ExpOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
  bool fastApproxMode = false;

  // Create query closure
  auto expOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::exp, device, inputSpec, fastApproxMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(expOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Binary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryEltwiseOpModel<OpTy>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
    llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

// Explicit template instantiation for BinaryEltwiseOpModel.
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::AddOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::MultiplyOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::SubtractOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::MaximumOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::MinimumOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::DivideOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::EqualOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::NotEqualOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::GreaterEqualOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::GreaterThanOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::LessEqualOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::LessThanOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::LogicalAndOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::LogicalOrOp>;
template struct BinaryEltwiseOpModel<mlir::tt::ttnn::LogicalXorOp>;

//===----------------------------------------------------------------------===//
// Ternary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> TernaryEltwiseOpModel<OpTy>::getOpConstraints(
    ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutC,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
    llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutC,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
template struct TernaryEltwiseOpModel<mlir::tt::ttnn::WhereOp>;

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> ReductionOpModel<OpTy>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
template struct ReductionOpModel<mlir::tt::ttnn::MeanOp>;
template struct ReductionOpModel<mlir::tt::ttnn::SumOp>;

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::SoftmaxOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     softmaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::SoftmaxOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(softmaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ReshapeOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ReshapeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
// SliceOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::SliceOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> begins,
    llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

  ::ttsl::Span<const int> beginsSpan = ::ttsl::make_span(beginsVec);
  ::ttsl::Span<const int> endsSpan = ::ttsl::make_span(endsVec);
  ::ttsl::Span<const int> stepSpan = ::ttsl::make_span(stepVec);

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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::SliceOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::ArrayRef<int64_t> begins,
    llvm::ArrayRef<int64_t> ends, llvm::ArrayRef<int64_t> step,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

  ::ttsl::Span<const int> beginsSpan = ::ttsl::make_const_span(beginsVec);
  ::ttsl::Span<const int> endsSpan = ::ttsl::make_const_span(endsVec);
  ::ttsl::Span<const int> stepSpan = ::ttsl::make_const_span(stepVec);

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
// TypecastOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::TypecastOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttcore::DataTypeAttr dtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::TypecastOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttcore::DataTypeAttr dtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ToLayoutOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> outputDtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ToLayoutOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> outputDtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ToMemoryConfigOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ToMemoryConfigOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    mlir::tt::ttnn::MemoryConfigAttr memoryConfig,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ConcatOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid,
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<mlir::tt::ttnn::TTNNLayoutAttr> inputLayouts, const int dim,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ConcatOp>::getOpRuntime(
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<mlir::tt::ttnn::TTNNLayoutAttr> inputLayouts, const int dim,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::TransposeOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::TransposeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
// LinearOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::LinearOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::LinearOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::MatmulOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::MatmulOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool transposeA,
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
// Conv2dOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::Conv2dOp>::getOpConstraints(
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
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::Conv2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
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
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ConvTranspose2dOp>::getOpConstraints(
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
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ConvTranspose2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
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
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
// MaxPool2D
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::MaxPool2dOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */);
  };

  return operation::getOpConstraints(inputLayout.getContext(), deviceGrid,
                                     maxPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::MaxPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */);
  };

  return operation::getOpRuntime(maxPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampScalar
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ClampScalarOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
    llvm::APFloat max, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::ClampScalarOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
    llvm::APFloat max, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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
// Permute
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::PermuteOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::PermuteOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::UpsampleOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::Attribute scaleFactor,
    llvm::StringRef mode, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::UpsampleOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::Attribute scaleFactor,
    llvm::StringRef mode, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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
                   mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                   llvm::ArrayRef<int64_t> weightShape,
                   mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
                   mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::EmbeddingOp>::getOpConstraints(
    mlir::tt::ttcore::GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
      inputLayout.isTiled() ? ::ttnn::TILE_LAYOUT : ::ttnn::ROW_MAJOR_LAYOUT;
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  ::ttnn::DataType dtype = conversion::getDataType(inputLayout.getDataType());

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

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::EmbeddingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
      inputLayout.isTiled() ? ::ttnn::TILE_LAYOUT : ::ttnn::ROW_MAJOR_LAYOUT;
  auto embeddingsType = ::ttnn::operations::embedding::EmbeddingsType::GENERIC;
  ::ttnn::DataType dtype = conversion::getDataType(inputLayout.getDataType());

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

} // namespace mlir::tt::op_model::ttnn
