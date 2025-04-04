// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"
#include <mlir/IR/MLIRContext.h>

#ifdef TTMLIR_ENABLE_OPMODEL

#include "Conversion.hpp"
#include "MetalHeaders.h"
#include "SingletonDeviceContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"

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
 * @brief Retrieves operation constraints based on the provided operation name
 * and callable.
 *
 * This function attempts to query operation constraints using the provided
 * callable and arguments. If successful, it returns a tuple with resource usage
 * details. Otherwise, an error message.
 *
 * @param name The name of the operation to query constraints for.
 * @param callable A callable object that performs the query.
 * @return A tuple containing query results or a string error.
 */
template <class Callable>
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
getOpConstraints(MLIRContext *context, std::string_view name,
                 Callable &callable) {
  ::ttnn::graph::ConstraintQueryResponse query;
  try {
    query = callable();
  } catch (const std::exception &e) {
    // We expect that query will handle exceptions and set error message. If
    // not, we should not continue.
    // TODO(rpavlovicTT): This should be a TT_FATAL.
    std::cerr << "Exception thrown during op constraints query: " << e.what()
              << std::endl;
    assert(false && "Exception thrown during op constraints query");
  }

  // Check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        query.error_message.value_or("<error message not set>"));
  }

  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core,
                         conversion::getLayoutAttrFromTensorSpec(
                             context, query.output_tensor_spec));
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
 * @brief Checks if the shard bounding box fits within the available grid size.
 *
 * This function verifies whether the shard bounding box specified in the
 * memory configuration fits within the range of device worker cores. If the
 * memory configuration is sharded and the shard bounding box exceeds the
 * available grid size, it throws a runtime error.
 *
 * @param computeGridSize The compute grid size.
 * @param memoryConfig The memory configuration which may specify a shard.
 *
 * @throws std::runtime_error If the shard bounding box is larger than the
 * available grid size.
 */
void checkGrid(const ::tt::tt_metal::CoreCoord &computeGridSize,
               const ::tt::tt_metal::MemoryConfig &memoryConfig) {
  if (memoryConfig.is_sharded()) {
    ::tt::tt_metal::CoreRange shardBoundingBox =
        memoryConfig.shard_spec.value().grid.bounding_box();
    ::tt::tt_metal::CoreRangeSet deviceWorkerCores{::tt::tt_metal::CoreRange{
        ::tt::tt_metal::CoreCoord{0, 0},
        ::tt::tt_metal::CoreCoord{computeGridSize.x - 1,
                                  computeGridSize.y - 1}}};
    if (deviceWorkerCores.contains(shardBoundingBox) == false) {
      throw std::runtime_error(
          "Selected shard is larger than available grid "
          "size. Compute Grid Size: " +
          computeGridSize.str() +
          ", selected bounding box: " + shardBoundingBox.str());
    }
  }
}

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
 * @brief Convenience wrapper to convert tuples of {shape, layout} into
 * TensorSpec. Validates worker grid size
 *
 * @param device Pointer to an open device to obtain the compute grid size
 * @param args 2-tuples of shape and layout
 */
template <typename... Args,
          typename = std::enable_if_t<
              (std::is_same_v<std::decay_t<Args>,
                              std::tuple<::llvm::ArrayRef<int64_t>,
                                         ::mlir::tt::ttnn::TTNNLayoutAttr>> &&
               ...)>>
auto convertToTensorSpec(::tt::tt_metal::IDevice *device, Args &&...args) {
  auto transformArg = [device](auto &&arg) {
    const ::ttnn::TensorSpec spec =
        conversion::getTensorSpec(std::get<0>(arg), std::get<1>(arg));
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      spec.memory_config());
    return spec;
  };
  return std::make_tuple(transformArg(std::forward<Args>(args))...);
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
                             llvm::ArrayRef<int64_t> inputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                             llvm::ArrayRef<int64_t> outputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto query = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        opSymbol, device, inputSpec,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(outputLayout.getContext(), opName, query);
}

template <typename OpSymbol>
llvm::Expected<size_t>
getEltwiseUnaryOpRuntime(std::string_view opName, OpSymbol opSymbol,
                         llvm::ArrayRef<int64_t> inputShape,
                         mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                         llvm::ArrayRef<int64_t> outputShape,
                         mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto query = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        opSymbol, device, inputSpec,
        outputSpec.tensor_layout().get_memory_config());
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
                              llvm::ArrayRef<int64_t> inputShapeA,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                              llvm::ArrayRef<int64_t> inputShapeB,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                              llvm::ArrayRef<int64_t> outputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShapeA, inputLayoutA),
      std::make_tuple(inputShapeB, inputLayoutB),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto query = [=]() {
    const auto [inputSpecA, inputSpecB, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        opSymbol, device, inputSpecA, inputSpecB, outputSpec.data_type(),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(outputLayout.getContext(), opName, query);
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
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShapeA, inputLayoutA),
      std::make_tuple(inputShapeB, inputLayoutB),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto query = [=]() {
    const auto [inputSpecA, inputSpecB, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        opSymbol, device, inputSpecA, inputSpecB, outputSpec.data_type(),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime(opName, query);
}
#endif

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
ReluOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseUnaryOpConstraints("ReluOpInterface", ::ttnn::relu,
                                      inputShape, inputLayout, outputShape,
                                      outputLayout);
#else
  return std::make_tuple(0, 0, 0);
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
SqrtOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseUnaryOpConstraints("SqrtOpInterface", ::ttnn::sqrt,
                                      inputShape, inputLayout, outputShape,
                                      outputLayout);
#else
  return std::make_tuple(0, 0, 0);
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
// AddOp
//===----------------------------------------------------------------------===//
llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
AddOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                 llvm::ArrayRef<int64_t> inputShapeB,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                 llvm::ArrayRef<int64_t> outputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseBinaryOpConstraints("AddOpInterface", ::ttnn::add,
                                       inputShapeA, inputLayoutA, inputShapeB,
                                       inputLayoutB, outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0);
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto softmaxOpQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::softmax, device, inputSpec, dimArg,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "SoftmaxOpInterface", softmaxOpQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
SoftmaxOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                 const int dimArg,
                                 llvm::ArrayRef<int64_t> outputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto softmaxOpQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::softmax, device, inputSpec, dimArg,
        outputSpec.tensor_layout().get_memory_config());
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
MeanOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  std::optional<llvm::ArrayRef<int64_t>> dimArg,
                                  bool keepDim,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout));
  auto memConfig = conversion::getMemoryConfig(outputLayout);

  std::optional<::ttnn::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto meanOpQuery = [=]() {
    const auto [inputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::mean, device, inputSpec, dimArgConverted, keepDim, memConfig);
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "MeanOpInterface", meanOpQuery);
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
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout));
  auto memConfig = conversion::getMemoryConfig(outputLayout);

  std::optional<::ttnn::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto meanOpQuery = [=]() {
    auto [inputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(::ttnn::mean, device, inputSpec,
                                           dimArgConverted, keepDim, memConfig);
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto reshapeOpQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::reshape, device, inputSpec, conversion::getShape(outputShape),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "ReshapeOpInterface", reshapeOpQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
ReshapeOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                 llvm::ArrayRef<int64_t> outputShape,
                                 mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto reshapeOpQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::reshape, device, inputSpec, conversion::getShape(outputShape),
        outputSpec.tensor_layout().get_memory_config());
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::tt::DataTypeAttr dtype,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto typecastOpQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "typecastOpInterface", typecastOpQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
TypecastOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  mlir::tt::DataTypeAttr dtype,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto typecastOpQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        outputSpec.tensor_layout().get_memory_config());
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::DataType> outputDtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool passDevicePtr) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout));

  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype) {
    dtype = conversion::getDataType(outputDtype.value());
  } else {
    dtype = std::nullopt;
  }

  std::optional<::tt::tt_metal::MemoryConfig> memoryConfig =
      std::make_optional(conversion::getMemoryConfig(outputLayout));

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    const auto [inputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        memoryConfig, passDevicePtr ? device : nullptr);
  };
  return operation::getOpConstraints(outputLayout.getContext(),
                                     "ToLayoutOpInterface", toLayoutOpQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
ToLayoutOpInterface::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  std::optional<mlir::tt::DataType> outputDtype,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                                  bool passDevicePtr) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout));

  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype) {
    dtype = conversion::getDataType(outputDtype.value());
  } else {
    dtype = std::nullopt;
  }

  std::optional<::tt::tt_metal::MemoryConfig> memoryConfig =
      std::make_optional(conversion::getMemoryConfig(outputLayout));

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    const auto [inputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        memoryConfig, passDevicePtr ? device : nullptr);
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout));

  // Create query closure
  auto transposeOpQuery = [=]() {
    const auto [inputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::transpose, device, inputSpec, dim0, dim1,
        conversion::getMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "TransposeOpInterface", transposeOpQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> TransposeOpInterface::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout));

  // Create query closure
  auto transposeOpQuery = [=]() {
    const auto [inputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::transpose, device, inputSpec, dim0, dim1,
        conversion::getMemoryConfig(outputLayout));
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
MatmulOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                    llvm::ArrayRef<int64_t> inputShapeB,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                    llvm::ArrayRef<int64_t> outputShape,
                                    mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                                    bool transposeA, bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShapeA, inputLayoutA),
      std::make_tuple(inputShapeB, inputLayoutB),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto matmulOpQuery = [=]() {
    const auto [inputSpecA, inputSpecB, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::matmul, device, inputSpecA, inputSpecB, transposeA, transposeB,
        outputSpec.tensor_layout().get_memory_config(), outputSpec.data_type());
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "MatmulOpInterface", matmulOpQuery);
#else
  return std::make_tuple(0, 0, 0);
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
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShapeA, inputLayoutA),
      std::make_tuple(inputShapeB, inputLayoutB),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto matmulOpQuery = [=]() {
    const auto [inputSpecA, inputSpecB, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::matmul, device, inputSpecA, inputSpecB, transposeA, transposeB,
        outputSpec.tensor_layout().get_memory_config(), outputSpec.data_type());
  };

  return operation::getOpRuntime("MatmulOpInterface", matmulOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<
    std::tuple<size_t, size_t, size_t, ::mlir::tt::ttnn::TTNNLayoutAttr>>
MultiplyOpInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  return getEltwiseBinaryOpConstraints("MultiplyOpInterface", ::ttnn::multiply,
                                       inputShapeA, inputLayoutA, inputShapeB,
                                       inputLayoutB, outputShape, outputLayout);
#else
  return std::make_tuple(0, 0, 0);
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
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
    int32_t in_channels, int32_t out_channels, int32_t batch_size,
    int32_t input_height, int32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    int32_t groups,
    std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(weightShape, weightLayout),
      std::make_tuple(outputShape, outputLayout));

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  auto conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    const auto [inputSpec, weightSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, biasTensor, conv2dConfigConverted, std::nullopt,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "Conv2dOpInterface", conv2dOpQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> Conv2dOpInterface::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape,
    mlir::tt::ttnn::TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<mlir::tt::ttnn::TTNNLayoutAttr> biasLayout,
    int32_t in_channels, int32_t out_channels, int32_t batch_size,
    int32_t input_height, int32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    int32_t groups,
    std::optional<mlir::tt::ttnn::Conv2dConfigAttr> conv2dConfig,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(weightShape, weightLayout),
      std::make_tuple(outputShape, outputLayout));

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  auto conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    const auto [inputSpec, weightSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, biasTensor, conv2dConfigConverted, std::nullopt,
        outputSpec.tensor_layout().get_memory_config());
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
MaxPool2DInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  // Prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto maxPool2DQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        outputSpec.tensor_layout().get_memory_config(),
        std::nullopt /* applied_shard_scheme */, ceilMode);
  };

  return operation::getOpConstraints(outputLayout.getContext(),
                                     "MaxPool2DInterface", maxPool2DQuery);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> MaxPool2DInterface::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, int32_t inputChannels,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool ceilMode, llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::IDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  // prepare io specs
  const auto specs = detail::convertToTensorSpec(
      device, std::make_tuple(inputShape, inputLayout),
      std::make_tuple(outputShape, outputLayout));

  // Create query closure
  auto maxPool2DQuery = [=]() {
    const auto [inputSpec, outputSpec] = specs;
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        outputSpec.tensor_layout().get_memory_config(),
        std::nullopt /* applied_shard_scheme */, ceilMode);
  };

  return operation::getOpRuntime("MaxPool2DInterface", maxPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
