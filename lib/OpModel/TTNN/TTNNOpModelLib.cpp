// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"
#include <type_traits>

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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
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
 * @param args Additional arguments to be forwarded to the callable.
 * @return A tuple containing query results or a string error.
 */
template <class Callable>
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getOpConstraints(std::string_view name, Callable &callable, auto &&...args) {
  ::ttnn::graph::ConstraintQueryResponse query;
  try {
    query = callable(std::forward<decltype(args)>(args)...);
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        query.error_message.value_or("<error message not set>"));
  }

  return std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                         query.resource_usage.l1_buffers_peak_per_core,
                         query.resource_usage.l1_output_buffer_per_core);
}

template <class Callable>
llvm::Expected<size_t> getOpRuntime(std::string_view name, Callable &callable,
                                    auto &&...args) {
  ::ttnn::graph::RuntimeQueryResponse query;
  try {
    query = callable(std::forward<decltype(args)>(args)...);
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
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
auto convertToTensorSpec(::tt::tt_metal::v0::IDevice *device, Args... args) {
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
// Template functions for binary elementwise operations.
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpSymbol>
llvm::Expected<std::tuple<size_t, size_t, size_t>>
getEltwiseBinaryOpConstraints(std::string_view opName, OpSymbol opSymbol,
                              llvm::ArrayRef<int64_t> inputShapeA,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                              llvm::ArrayRef<int64_t> inputShapeB,
                              mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                              llvm::ArrayRef<int64_t> outputShape,
                              mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
  auto query = [&](llvm::ArrayRef<int64_t> aShape,
                   mlir::tt::ttnn::TTNNLayoutAttr aLayout,
                   llvm::ArrayRef<int64_t> bShape,
                   mlir::tt::ttnn::TTNNLayoutAttr bLayout,
                   llvm::ArrayRef<int64_t> outShape,
                   mlir::tt::ttnn::TTNNLayoutAttr outLayout) {
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();
    const auto [inputSpecA, inputSpecB, outputSpec] =
        detail::convertToTensorSpec(device, std::make_tuple(aShape, aLayout),
                                    std::make_tuple(bShape, bLayout),
                                    std::make_tuple(outShape, outLayout));

    return ::ttnn::graph::query_op_constraints(
        opSymbol, device, inputSpecA, inputSpecB, outputSpec.data_type(),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints(opName, query, inputShapeA, inputLayoutA,
                                     inputShapeB, inputLayoutB, outputShape,
                                     outputLayout);
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
  auto query = [&](llvm::ArrayRef<int64_t> aShape,
                   mlir::tt::ttnn::TTNNLayoutAttr aLayout,
                   llvm::ArrayRef<int64_t> bShape,
                   mlir::tt::ttnn::TTNNLayoutAttr bLayout,
                   llvm::ArrayRef<int64_t> outShape,
                   mlir::tt::ttnn::TTNNLayoutAttr outLayout) {
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();
    const auto [inputSpecA, inputSpecB, outputSpec] =
        detail::convertToTensorSpec(device, std::make_tuple(aShape, aLayout),
                                    std::make_tuple(bShape, bLayout),
                                    std::make_tuple(outShape, outLayout));

    return ::ttnn::graph::query_op_runtime(
        opSymbol, device, inputSpecA, inputSpecB, outputSpec.data_type(),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime(opName, query, inputShapeA, inputLayoutA,
                                 inputShapeB, inputLayoutB, outputShape,
                                 outputLayout);
}
#endif

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
ReluOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  llvm::ArrayRef<int64_t> outputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto reluOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                        mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                        llvm::ArrayRef<int64_t> outputShape,
                        mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::relu, device, inputSpec,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("ReluOpInterface", reluOpQuery, inputShape,
                                     inputLayout, outputShape, outputLayout);
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
  auto reluOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                        mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                        llvm::ArrayRef<int64_t> outputShape,
                        mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::relu, device, inputSpec,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime("ReluOpInterface", reluOpQuery, inputShape,
                                 inputLayout, outputShape, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
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
llvm::Expected<std::tuple<size_t, size_t, size_t>>
SoftmaxOpInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dimArg,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto softmaxOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                           const int dimArg,
                           llvm::ArrayRef<int64_t> outputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::softmax, device, inputSpec, dimArg,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("SoftmaxOpInterface", softmaxOpQuery,
                                     inputShape, inputLayout, dimArg,
                                     outputShape, outputLayout);
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
  auto softmaxOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                           const int dimArg,
                           llvm::ArrayRef<int64_t> outputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::softmax, device, inputSpec, dimArg,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime("SoftmaxOpInterface", softmaxOpQuery,
                                 inputShape, inputLayout, dimArg, outputShape,
                                 outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
MeanOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                  mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                                  std::optional<llvm::ArrayRef<int64_t>> dimArg,
                                  bool keepDim,
                                  mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto meanOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                        mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                        std::optional<llvm::ArrayRef<int64_t>> dimArg,
                        bool keepDim,
                        mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    auto [inputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout));
    auto memConfig = conversion::getMemoryConfig(outputLayout);

    std::optional<::ttnn::SmallVector<int>> dimArgConverted;
    if (dimArg) {
      dimArgConverted =
          conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
    } else {
      dimArgConverted = std::nullopt;
    }
    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::mean, device, inputSpec, dimArgConverted, keepDim, memConfig);
  };

  return operation::getOpConstraints("MeanOpInterface", meanOpQuery, inputShape,
                                     inputLayout, dimArg, keepDim,
                                     outputLayout);
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
  auto meanOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                        mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                        std::optional<llvm::ArrayRef<int64_t>> dimArg,
                        bool keepDim,
                        mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    auto [inputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout));
    auto memConfig = conversion::getMemoryConfig(outputLayout);

    std::optional<::ttnn::SmallVector<int>> dimArgConverted;
    if (dimArg) {
      dimArgConverted =
          conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
    } else {
      dimArgConverted = std::nullopt;
    }

    // run op runtime query
    return ::ttnn::graph::query_op_runtime(::ttnn::mean, device, inputSpec,
                                           dimArgConverted, keepDim, memConfig);
  };

  return operation::getOpRuntime("MeanOpInterface", meanOpQuery, inputShape,
                                 inputLayout, dimArg, keepDim, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
ReshapeOpInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto reshapeOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                           llvm::ArrayRef<int64_t> outputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::reshape, device, inputSpec, conversion::getShape(outputShape),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("ReshapeOpInterface", reshapeOpQuery,
                                     inputShape, inputLayout, outputShape,
                                     outputLayout);
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
  auto reshapeOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                           llvm::ArrayRef<int64_t> outputShape,
                           mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::reshape, device, inputSpec, conversion::getShape(outputShape),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime("ReshapeOpInterface", reshapeOpQuery,
                                 inputShape, inputLayout, outputShape,
                                 outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
TypecastOpInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, mlir::tt::DataTypeAttr dtype,
    llvm::ArrayRef<int64_t> outputShape,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto typecastOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                            mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                            mlir::tt::DataTypeAttr dtype,
                            llvm::ArrayRef<int64_t> outputShape,
                            mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("typecastOpInterface", typecastOpQuery,
                                     inputShape, inputLayout, dtype,
                                     outputShape, outputLayout);
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
  auto typecastOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                            mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                            mlir::tt::DataTypeAttr dtype,
                            llvm::ArrayRef<int64_t> outputShape,
                            mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::typecast, device, inputSpec,
        conversion::getDataType(dtype.getValue()),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime("TypecastOpInterface", typecastOpQuery,
                                 inputShape, inputLayout, dtype, outputShape,
                                 outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
ToLayoutOpInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::DataType> outputDtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout, bool passDevicePtr) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto toLayoutOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                            mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                            std::optional<mlir::tt::DataType> outputDtype,
                            mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                            bool passDevicePtr) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    auto [inputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout));

    std::optional<::tt::tt_metal::DataType> dtype;
    if (outputDtype) {
      dtype = conversion::getDataType(outputDtype.value());
    } else {
      dtype = std::nullopt;
    }

    std::optional<::tt::tt_metal::MemoryConfig> memoryConfig =
        std::make_optional(conversion::getMemoryConfig(outputLayout));

    return ::ttnn::graph::query_op_constraints(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        memoryConfig, passDevicePtr ? device : nullptr);
  };

  return operation::getOpConstraints("ToLayoutOpInterface", toLayoutOpQuery,
                                     inputShape, inputLayout, outputDtype,
                                     outputLayout, passDevicePtr);
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
  auto toLayoutOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                            mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                            std::optional<mlir::tt::DataType> outputDtype,
                            mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                            bool passDevicePtr) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    auto [inputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout));

    std::optional<::tt::tt_metal::DataType> dtype;
    if (outputDtype) {
      dtype = conversion::getDataType(outputDtype.value());
    } else {
      dtype = std::nullopt;
    }

    std::optional<::tt::tt_metal::MemoryConfig> memoryConfig =
        std::make_optional(conversion::getMemoryConfig(outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        memoryConfig, passDevicePtr ? device : nullptr);
  };

  return operation::getOpRuntime("ToLayoutOpInterface", toLayoutOpQuery,
                                 inputShape, inputLayout, outputDtype,
                                 outputLayout, passDevicePtr);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
TransposeOpInterface::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto transposeOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                             const int dim0, const int dim1,
                             mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::transpose, device, inputSpec, dim0, dim1,
        conversion::getMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints("TransposeOpInterface", transposeOpQuery,
                                     inputShape, inputLayout, dim0, dim1,
                                     outputLayout);
#else
  return std::make_tuple(0, 0, 0);
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> TransposeOpInterface::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape,
    mlir::tt::ttnn::TTNNLayoutAttr inputLayout, const int dim0, const int dim1,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto transposeOpQuery = [](llvm::ArrayRef<int64_t> inputShape,
                             mlir::tt::ttnn::TTNNLayoutAttr inputLayout,
                             const int dim0, const int dim1,
                             mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::transpose, device, inputSpec, dim0, dim1,
        conversion::getMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime("TransposeOpInterface", transposeOpQuery,
                                 inputShape, inputLayout, dim0, dim1,
                                 outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
llvm::Expected<std::tuple<size_t, size_t, size_t>>
MatmulOpInterface::getOpConstraints(llvm::ArrayRef<int64_t> inputShapeA,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                                    llvm::ArrayRef<int64_t> inputShapeB,
                                    mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                                    llvm::ArrayRef<int64_t> outputShape,
                                    mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                                    bool transposeA, bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto matmulOpQuery = [](llvm::ArrayRef<int64_t> inputShapeA,
                          mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                          llvm::ArrayRef<int64_t> inputShapeB,
                          mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                          llvm::ArrayRef<int64_t> outputShape,
                          mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                          bool transposeA, bool transposeB) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpecA, inputSpecB, outputSpec] =
        detail::convertToTensorSpec(device,
                                    std::make_tuple(inputShapeA, inputLayoutA),
                                    std::make_tuple(inputShapeB, inputLayoutB),
                                    std::make_tuple(outputShape, outputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::matmul, device, inputSpecA, inputSpecB, transposeA, transposeB,
        outputSpec.tensor_layout().get_memory_config(), outputSpec.data_type());
  };

  return operation::getOpConstraints("MatmulOpInterface", matmulOpQuery,
                                     inputShapeA, inputLayoutA, inputShapeB,
                                     inputLayoutB, outputShape, outputLayout,
                                     transposeA, transposeB);
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
  auto matmulOpQuery = [](llvm::ArrayRef<int64_t> inputShapeA,
                          mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA,
                          llvm::ArrayRef<int64_t> inputShapeB,
                          mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB,
                          llvm::ArrayRef<int64_t> outputShape,
                          mlir::tt::ttnn::TTNNLayoutAttr outputLayout,
                          bool transposeA, bool transposeB) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpecA, inputSpecB, outputSpec] =
        detail::convertToTensorSpec(device,
                                    std::make_tuple(inputShapeA, inputLayoutA),
                                    std::make_tuple(inputShapeB, inputLayoutB),
                                    std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::matmul, device, inputSpecA, inputSpecB, transposeA, transposeB,
        outputSpec.tensor_layout().get_memory_config(), outputSpec.data_type());
  };

  return operation::getOpRuntime("MatmulOpInterface", matmulOpQuery,
                                 inputShapeA, inputLayoutA, inputShapeB,
                                 inputLayoutB, outputShape, outputLayout,
                                 transposeA, transposeB);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<std::tuple<size_t, size_t, size_t>>
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
llvm::Expected<std::tuple<size_t, size_t, size_t>>
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
  auto conv2dOpQuery =
      [](llvm::ArrayRef<int64_t> inputShape,
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
        // open device device, will close it at the end of function

        ::tt::tt_metal::v0::IDevice *device =
            SingletonDeviceContext::getInstance().getDevice();

        // prepare io specs
        const auto [inputSpec, weightSpec, outputSpec] =
            detail::convertToTensorSpec(
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

        return ::ttnn::graph::query_op_constraints(
            ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
            out_channels, batch_size, input_height, input_width,
            conversion::convertArrayRefToArray(kernel_size),
            conversion::convertArrayRefToArray(stride),
            conversion::convertArrayRefToArray(padding),
            conversion::convertArrayRefToArray(dilation), groups, biasTensor,
            conv2dConfigConverted, std::nullopt,
            outputSpec.tensor_layout().get_memory_config());
      };

  return operation::getOpConstraints(
      "Conv2dOpInterface", conv2dOpQuery, inputShape, inputLayout, weightShape,
      weightLayout, biasShape, biasLayout, in_channels, out_channels,
      batch_size, input_height, input_width, kernel_size, stride, padding,
      dilation, groups, conv2dConfig, outputShape, outputLayout);
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
  auto conv2dOpQuery =
      [](llvm::ArrayRef<int64_t> inputShape,
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
        // open device device, will close it at the end of function
        ::tt::tt_metal::v0::IDevice *device =
            SingletonDeviceContext::getInstance().getDevice();

        // prepare io specs
        const auto [inputSpec, weightSpec, outputSpec] =
            detail::convertToTensorSpec(
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

        return ::ttnn::graph::query_op_runtime(
            ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
            out_channels, batch_size, input_height, input_width,
            conversion::convertArrayRefToArray(kernel_size),
            conversion::convertArrayRefToArray(stride),
            conversion::convertArrayRefToArray(padding),
            conversion::convertArrayRefToArray(dilation), groups, biasTensor,
            conv2dConfigConverted, std::nullopt,
            outputSpec.tensor_layout().get_memory_config());
      };

  return operation::getOpRuntime(
      "Conv2dOpInterface", conv2dOpQuery, inputShape, inputLayout, weightShape,
      weightLayout, biasShape, biasLayout, in_channels, out_channels,
      batch_size, input_height, input_width, kernel_size, stride, padding,
      dilation, groups, conv2dConfig, outputShape, outputLayout);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
