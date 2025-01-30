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
 * callable and arguments. It returns a tuple containing a boolean indicating
 * success or failure, an optional tuple with resource usage details (if
 * successful), and an optional error message (if failed).
 *
 * @param name The name of the operation to query constraints for.
 * @param callable A callable object that performs the query.
 * @param args Additional arguments to be forwarded to the callable.
 * @return A tuple containing query results.
 */
template <class Callable>
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
getOpConstraints(const std::string_view &name, Callable &callable,
                 auto &&...args) {
  ::ttnn::graph::ConstraintQueryResponse query;
  try {
    query = callable(std::forward<decltype(args)>(args)...);
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return std::make_tuple(
        false, std::nullopt,
        query.error_message.value_or("<error message not set>"));
  }

  return std::make_tuple(
      true,
      std::make_tuple(query.resource_usage.cb_peak_size_per_core,
                      query.resource_usage.l1_buffers_peak_per_core,
                      query.resource_usage.l1_output_buffer_per_core),
      std::nullopt);
}

template <class Callable>
std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
getOpRuntime(const std::string_view &name, Callable &callable, auto &&...args) {
  ::ttnn::graph::RuntimeQueryResponse query;
  try {
    query = callable(std::forward<decltype(args)>(args)...);
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return std::make_tuple(
        false, std::nullopt,
        query.error_message.value_or("<error message not set>"));
  }

  return std::make_tuple(true, query.runtime, std::nullopt);
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
               const mlir::tt::GridAttr &workerGrid) {
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

std::tuple<bool, std::optional<std::string>>
Device::getDeviceConstraints(const mlir::tt::GridAttr &workerGrid) {
#ifdef TTMLIR_ENABLE_OPMODEL
  try {
    detail::checkGrid(SingletonDeviceContext::getInstance()
                          .getDevice()
                          ->compute_with_storage_grid_size(),
                      workerGrid);
  } catch (const std::exception &e) {
    return std::make_tuple(false, e.what());
  }
#endif
  return std::make_tuple(true, std::nullopt);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
ReluOpInterface::getOpConstraints(
    const ::llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto reluOpQuery = [](const ::llvm::ArrayRef<int64_t> &inputShape,
                        const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
                        const ::llvm::ArrayRef<int64_t> &outputShape,
                        const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
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
  return std::make_tuple(false, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
ReluOpInterface::getOpRuntime(
    const ::llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto reluOpQuery = [](const ::llvm::ArrayRef<int64_t> &inputShape,
                        const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
                        const ::llvm::ArrayRef<int64_t> &outputShape,
                        const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
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
  return std::make_tuple(false, 0, std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
AddOpInterface::getOpConstraints(
    const ::llvm::ArrayRef<int64_t> &inputShapeA,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
    const ::llvm::ArrayRef<int64_t> &inputShapeB,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto addOpQuery = [](const ::llvm::ArrayRef<int64_t> &inputShapeA,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
                       const ::llvm::ArrayRef<int64_t> &inputShapeB,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
                       const ::llvm::ArrayRef<int64_t> &outputShape,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpecA, inputSpecB, outputSpec] =
        detail::convertToTensorSpec(device,
                                    std::make_tuple(inputShapeA, inputLayoutA),
                                    std::make_tuple(inputShapeB, inputLayoutB),
                                    std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_constraints(
        ::ttnn::add, device, inputSpecA, inputSpecB, outputSpec.data_type(),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("AddOpInterface", addOpQuery, inputShapeA,
                                     inputLayoutA, inputShapeB, inputLayoutB,
                                     outputShape, outputLayout);
#else
  return std::make_tuple(false, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
AddOpInterface::getOpRuntime(
    const ::llvm::ArrayRef<int64_t> &inputShapeA,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
    const ::llvm::ArrayRef<int64_t> &inputShapeB,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto addOpQuery = [](const ::llvm::ArrayRef<int64_t> &inputShapeA,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
                       const ::llvm::ArrayRef<int64_t> &inputShapeB,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
                       const ::llvm::ArrayRef<int64_t> &outputShape,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
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
        ::ttnn::add, device, inputSpecA, inputSpecB, outputSpec.data_type(),
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime("AddOpInterface", addOpQuery, inputShapeA,
                                 inputLayoutA, inputShapeB, inputLayoutB,
                                 outputShape, outputLayout);
#else
  return std::make_tuple(false, 0, std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
SoftmaxOpInterface::getOpConstraints(
    const llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto softmaxOpQuery = [](const llvm::ArrayRef<int64_t> &inputShape,
                           const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
                           const int dim_arg,
                           const llvm::ArrayRef<int64_t> &outputShape,
                           const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::softmax, device, inputSpec, dim_arg,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("SoftmaxOpInterface", softmaxOpQuery,
                                     inputShape, inputLayout, dim_arg,
                                     outputShape, outputLayout);
#else
  return std::make_tuple(false, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
SoftmaxOpInterface::getOpRuntime(
    const llvm::ArrayRef<int64_t> &inputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout, const int dim_arg,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto softmaxOpQuery = [](const llvm::ArrayRef<int64_t> &inputShape,
                           const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout,
                           const int dim_arg,
                           const llvm::ArrayRef<int64_t> &outputShape,
                           const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::IDevice *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const auto [inputSpec, outputSpec] = detail::convertToTensorSpec(
        device, std::make_tuple(inputShape, inputLayout),
        std::make_tuple(outputShape, outputLayout));

    return ::ttnn::graph::query_op_runtime(
        ::ttnn::softmax, device, inputSpec, dim_arg,
        outputSpec.tensor_layout().get_memory_config());
  };

  return operation::getOpRuntime("SoftmaxOpInterface", softmaxOpQuery,
                                 inputShape, inputLayout, dim_arg, outputShape,
                                 outputLayout);
#else
  return std::make_tuple(false, 0, std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
MatmulOpInterface::getOpConstraints(
    const llvm::ArrayRef<int64_t> &inputShapeA,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
    const llvm::ArrayRef<int64_t> &inputShapeB,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transposeA,
    bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto matmulOpQuery = [](const llvm::ArrayRef<int64_t> &inputShapeA,
                          const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
                          const llvm::ArrayRef<int64_t> &inputShapeB,
                          const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
                          const llvm::ArrayRef<int64_t> &outputShape,
                          const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
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
  return std::make_tuple(false, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

std::tuple<bool, std::optional<size_t>, std::optional<std::string>>
MatmulOpInterface::getOpRuntime(
    const llvm::ArrayRef<int64_t> &inputShapeA,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
    const llvm::ArrayRef<int64_t> &inputShapeB,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transposeA,
    bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto matmulOpQuery = [](const llvm::ArrayRef<int64_t> &inputShapeA,
                          const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutA,
                          const llvm::ArrayRef<int64_t> &inputShapeB,
                          const mlir::tt::ttnn::TTNNLayoutAttr &inputLayoutB,
                          const llvm::ArrayRef<int64_t> &outputShape,
                          const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
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
  return std::make_tuple(false, 0, std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
