// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/IR/Utils.h"
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
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(std::string_view name, MLIRContext *context,
                 GridAttr deviceGrid, Callable &callable) {

  llvm::Expected<::ttnn::graph::ConstraintQueryResponse> query =
      executeConstraintQuery<Callable>(callable);
  if (auto error = query.takeError()) {
    return error;
  }

  ::ttnn::graph::ConstraintQueryResponse response = query.get();

  return std::make_tuple(
      response.resource_usage.cb_peak_size_per_core,
      response.resource_usage.l1_buffers_peak_per_core,
      response.resource_usage.l1_output_buffer_per_core,
      conversion::getLayoutAttrFromTensorSpec(
          context, response.output_tensor_spec.value(), deviceGrid.getShape()));
}

template <class Callable>
llvm::Expected<size_t> getOpRuntime(std::string_view name, Callable &callable) {
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
               mlir::tt::GridAttr workerGrid) {
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

} // namespace detail
#endif // TTMLIR_ENABLE_OPMODEL

bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr layout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Conversion to TensorSpec may throw if the layout is invalid, in which case
  // we return false.
  try {
    conversion::getTensorSpec(tensorShape, layout);
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
                      ::mlir::tt::DataType dataType) {
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
    std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig,
    bool hasBias) {
  if (weightLayout.getBufferType() !=
      mlir::tt::ttnn::BufferType::SystemMemory) {
    llvm::report_fatal_error("Conv2d weight tensor assumed to be on host.");
  }

  // Create ttnn weight tesnor.
  //
  // TODO(#3070): Prepare conv2d weights only works with host tesnsors. This
  // is slow and undesireable. We will move this to device once change
  // https://github.com/tenstorrent/tt-metal/issues/20503 on metal lands in
  // tt-mlir.
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

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  auto prepare_fn = &::ttnn::operations::conv::conv2d::prepare_conv_weights<
      ::tt::tt_metal::distributed::MeshDevice>;
  // Create query closure
  auto prepareConv2dWeightsOpQuery = [=]() {
    ::ttnn::operations::conv::conv2d::Conv2dConfig localConfig;
    if (!conv2dConfigConverted.has_value()) {
      localConfig = ::ttnn::operations::conv::conv2d::Conv2dConfig();
      // TODO(#2441): Need to match tensor dtypes with conv2d config.
      // This will be fixed on IR side shortly.
      localConfig.dtype = inputSpec.data_type();
      localConfig.weights_dtype = weightTensor.dtype();
    } else {
      localConfig = *conv2dConfigConverted;
    }

    return ::ttnn::graph::query_op_constraints(
        prepare_fn, device, weightTensor, inputSpec.memory_config(),
        inputSpec.layout(), "OIHW", in_channels, out_channels, batch_size,
        input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, localConfig, std::nullopt, std::nullopt);
  };

  auto output = operation::executeConstraintQuery(prepareConv2dWeightsOpQuery);

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
          op->getConv2dConfig(), op->getBias() != nullptr);
  if (!outputTensorSpec) {
    llvm::errs() << llvm::toString(outputTensorSpec.takeError());
    assert(false && "Failed to calculate conv2d prepared weights shape.");
  }

  // Convert back to RankedTensorType
  auto deviceGrid =
      mlir::tt::lookupDevice(op->getOperation()).getWorkerGrid().getShape();

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
Device::getDeviceConstraints(mlir::tt::GridAttr workerGrid) {
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
// Template functions for unary elementwise operations.
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpSymbol>
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getEltwiseUnaryOpConstraints(std::string_view opName, OpSymbol opSymbol,
                             GridAttr deviceGrid,
                             llvm::ArrayRef<int64_t> inputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                             llvm::ArrayRef<int64_t> outputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
        opSymbol, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(opName, inputLayout.getContext(),
                                     deviceGrid, query);
}

template <typename OpSymbol>
llvm::Expected<size_t>
getEltwiseUnaryOpRuntime(std::string_view opName, OpSymbol opSymbol,
                         llvm::ArrayRef<int64_t> inputShape,
                         mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                         llvm::ArrayRef<int64_t> outputShape,
                         mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
        opSymbol, device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(opName, query);
}
#endif

//===----------------------------------------------------------------------===//
// Template functions for binary elementwise operations.
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpSymbol>
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getEltwiseBinaryOpConstraints(std::string_view opName, OpSymbol opSymbol,
                              GridAttr deviceGrid,
                              llvm::ArrayRef<int64_t> inputShapeA,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                              llvm::ArrayRef<int64_t> inputShapeB,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                              llvm::ArrayRef<int64_t> outputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

  std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt;
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig = std::nullopt;
  if (outputLayout) {
    auto outputSpecExp =
        detail::convertToTensorSpec(device, outputShape, outputLayout);
    if (!outputSpecExp) {
      return outputSpecExp.takeError();
    }
    ::ttnn::TensorSpec outputSpec = outputSpecExp.get();
    outputDType = outputSpec.data_type();
    outputMemoryConfig = outputSpec.memory_config();
  }

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_constraints(opSymbol, device, inputSpecA,
                                               inputSpecB, outputDType,
                                               outputMemoryConfig);
  };

  return operation::getOpConstraints(opName, inputLayoutA.getContext(),
                                     deviceGrid, query);
}

template <typename OpSymbol>
llvm::Expected<size_t>
getEltwiseBinaryOpRuntime(std::string_view opName, OpSymbol opSymbol,
                          llvm::ArrayRef<int64_t> inputShapeA,
                          mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                          llvm::ArrayRef<int64_t> inputShapeB,
                          mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                          llvm::ArrayRef<int64_t> outputShape,
                          mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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

  std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt;
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig = std::nullopt;
  if (outputLayout) {
    auto outputSpecExp =
        detail::convertToTensorSpec(device, outputShape, outputLayout);
    if (!outputSpecExp) {
      return outputSpecExp.takeError();
    }
    ::ttnn::TensorSpec outputSpec = outputSpecExp.get();
    outputDType = outputSpec.data_type();
    outputMemoryConfig = outputSpec.memory_config();
  }
  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(opSymbol, device, inputSpecA,
                                           inputSpecB, outputDType,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(opName, query);
}
#endif

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ReluOpInterface::getOpConstraints(GridAttr deviceGrid,
                                  llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseUnaryOpConstraints("ReluOpInterface", ::ttnn::relu,
                                      deviceGrid, inputShape, inputLayout,
                                      outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
ReluOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                              llvm::ArrayRef<int64_t> outputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseUnaryOpRuntime("ReluOpInterface", ::ttnn::relu, inputShape,
                                  inputLayout, outputShape, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SqrtOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
SqrtOpInterface::getOpConstraints(GridAttr deviceGrid,
                                  llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseUnaryOpConstraints("SqrtOpInterface", ::ttnn::sqrt,
                                      deviceGrid, inputShape, inputLayout,
                                      outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
SqrtOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                              llvm::ArrayRef<int64_t> outputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseUnaryOpRuntime("SqrtOpInterface", ::ttnn::sqrt, inputShape,
                                  inputLayout, outputShape, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
SigmoidOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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

  return operation::getOpConstraints(
      "SigmoidOpInterface", inputLayout.getContext(), deviceGrid, query);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
SigmoidOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
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

  return operation::getOpRuntime("SigmoidOpInterface", query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
AddOpInterface::getOpConstraints(GridAttr deviceGrid,
                                 llvm::ArrayRef<int64_t> inputShapeA,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                 llvm::ArrayRef<int64_t> inputShapeB,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                 llvm::ArrayRef<int64_t> outputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseBinaryOpConstraints(
      "AddOpInterface", ::ttnn::add, deviceGrid, inputShapeA, inputLayoutA,
      inputShapeB, inputLayoutB, outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
AddOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
                             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                             llvm::ArrayRef<int64_t> inputShapeB,
                             mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                             llvm::ArrayRef<int64_t> outputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseBinaryOpRuntime("AddOpInterface", ::ttnn::add, inputShapeA,
                                   inputLayoutA, inputShapeB, inputLayoutB,
                                   outputShape, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
SoftmaxOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
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
  auto softmaxOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::softmax, device, inputSpec, dimArg,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints("SoftmaxOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     softmaxOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
SoftmaxOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                 const int dimArg,
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
  auto softmaxOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::softmax, device, inputSpec, dimArg,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime("SoftmaxOpInterface", softmaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MeanOpInterface::getOpConstraints(GridAttr deviceGrid,
                                  llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  std::optional<llvm::ArrayRef<int64_t>> dimArg,
                                  bool keepDim,
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

  std::optional<::ttnn::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto meanOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::mean, device, inputSpec, dimArgConverted, keepDim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(
      "MeanOpInterface", inputLayout.getContext(), deviceGrid, meanOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
MeanOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                              std::optional<llvm::ArrayRef<int64_t>> dimArg,
                              bool keepDim,
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

  std::optional<::ttnn::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto meanOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::mean, device, inputSpec, dimArgConverted, keepDim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime("MeanOpInterface", meanOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ReshapeOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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

  return operation::getOpConstraints("ReshapeOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     reshapeOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
ReshapeOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
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

  return operation::getOpRuntime("ReshapeOpInterface", reshapeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
TypecastOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::tt::DataTypeAttr dtype,
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
  auto typecastOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints("typecastOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     typecastOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
TypecastOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  mlir::tt::DataTypeAttr dtype,
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
  auto typecastOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime("TypecastOpInterface", typecastOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ToLayoutOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::DataType> outputDtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool passDevicePtr) {
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
        detail::getNullableMemoryConfig(outputLayout),
        passDevicePtr ? device : nullptr);
  };
  return operation::getOpConstraints("ToLayoutOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     toLayoutOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
ToLayoutOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  std::optional<mlir::tt::DataType> outputDtype,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                                  bool passDevicePtr) {
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
        detail::getNullableMemoryConfig(outputLayout),
        passDevicePtr ? device : nullptr);
  };

  return operation::getOpRuntime("ToLayoutOpInterface", toLayoutOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
TransposeOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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

  return operation::getOpConstraints("TransposeOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     transposeOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> TransposeOpInterface::getOpRuntime(
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

  return operation::getOpRuntime("TransposeOpInterface", transposeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MatmulOpInterface::getOpConstraints(GridAttr deviceGrid,
                                    llvm::ArrayRef<int64_t> inputShapeA,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                    llvm::ArrayRef<int64_t> inputShapeB,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                    llvm::ArrayRef<int64_t> outputShape,
                                    mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
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

  std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt;
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig = std::nullopt;
  if (outputLayout) {
    auto outputSpecExp =
        detail::convertToTensorSpec(device, outputShape, outputLayout);
    if (!outputSpecExp) {
      return outputSpecExp.takeError();
    }
    ::ttnn::TensorSpec outputSpec = outputSpecExp.get();
    outputDType = outputSpec.data_type();
    outputMemoryConfig = outputSpec.memory_config();
  }

  // Create query closure
  auto matmulOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::matmul, device, inputSpecA, inputSpecB, transposeA, transposeB,
        outputMemoryConfig, outputDType);
  };

  return operation::getOpConstraints("MatmulOpInterface",
                                     inputLayoutA.getContext(), deviceGrid,
                                     matmulOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
MatmulOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
                                mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                llvm::ArrayRef<int64_t> inputShapeB,
                                mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                llvm::ArrayRef<int64_t> outputShape,
                                mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
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

  std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt;
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig = std::nullopt;
  if (outputLayout) {
    auto outputSpecExp =
        detail::convertToTensorSpec(device, outputShape, outputLayout);
    if (!outputSpecExp) {
      return outputSpecExp.takeError();
    }
    ::ttnn::TensorSpec outputSpec = outputSpecExp.get();
    outputDType = outputSpec.data_type();
    outputMemoryConfig = outputSpec.memory_config();
  }

  // Create query closure
  auto matmulOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(::ttnn::matmul, device, inputSpecA,
                                           inputSpecB, transposeA, transposeB,
                                           outputMemoryConfig, outputDType);
  };

  return operation::getOpRuntime("MatmulOpInterface", matmulOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MultiplyOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseBinaryOpConstraints(
      "MultiplyOpInterface", ::ttnn::multiply, deviceGrid, inputShapeA,
      inputLayoutA, inputShapeB, inputLayoutB, outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
MultiplyOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShapeA,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                  llvm::ArrayRef<int64_t> inputShapeB,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseBinaryOpRuntime("MultiplyOpInterface", ::ttnn::multiply,
                                   inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, outputShape, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
Conv2dOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
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
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig,
          biasLayout.has_value());
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

  std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
      conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    ::ttnn::operations::conv::conv2d::Conv2dConfig localConfig;
    if (!conv2dConfigConverted.has_value()) {
      localConfig = ::ttnn::operations::conv::conv2d::Conv2dConfig();
      // TODO(#2441): Need to match tensor dtypes with conv2d config.
      // This will be fixed on IR side shortly.
      localConfig.dtype = inputSpec.data_type();
      localConfig.weights_dtype = weightSpec.data_type();
    } else {
      localConfig = *conv2dConfigConverted;
    }

    return ::ttnn::graph::query_op_constraints(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, biasTensor, localConfig, std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(
      "Conv2dOpInterface", inputLayout.getContext(), deviceGrid, conv2dOpQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> Conv2dOpInterface::getOpRuntime(
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
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL

  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig,
          biasLayout.has_value());
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

  auto conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    ::ttnn::operations::conv::conv2d::Conv2dConfig localConfig;
    if (!conv2dConfigConverted.has_value()) {
      localConfig = ::ttnn::operations::conv::conv2d::Conv2dConfig();
      // TODO(#2441): Need to match tensor dtypes with conv2d config.
      // This will be fixed on IR side shortly.
      localConfig.dtype = inputSpec.data_type();
      localConfig.weights_dtype = weightSpec.data_type();
    } else {
      localConfig = *conv2dConfigConverted;
    }

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, biasTensor, localConfig, std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime("Conv2dOpInterface", conv2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MaxPool2D
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MaxPool2DOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */, ceilMode);
  };

  return operation::getOpConstraints("MaxPool2DOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     maxPool2DQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> MaxPool2DOpInterface::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
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
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* applied_shard_scheme */, ceilMode);
  };

  return operation::getOpRuntime("MaxPool2DOpInterface", maxPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampScalar
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ClampScalarOpInterface::getOpConstraints(
    GridAttr deviceGrid, llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
    llvm::APFloat max, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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

  return operation::getOpConstraints("ClampScalarOpInterface",
                                     inputLayout.getContext(), deviceGrid,
                                     clampScalarQuery);
#else
  return std::make_tuple(0, 0, 0, nullptr);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> ClampScalarOpInterface::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, llvm::APFloat min,
    llvm::APFloat max, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

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

  return operation::getOpRuntime("ClampScalarOpInterface", clampScalarQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
