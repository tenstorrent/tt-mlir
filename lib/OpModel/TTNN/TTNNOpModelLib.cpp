// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"

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
  ::ttnn::graph::QueryResponse query;
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
    ::tt::tt_metal::v0::Device *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const ::ttnn::TensorSpec input_spec =
        conversion::getTensorSpec(inputShape, inputLayout);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      input_spec.memory_config());
    const ::ttnn::TensorSpec output_spec =
        conversion::getTensorSpec(outputShape, outputLayout);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      output_spec.memory_config());

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::relu, device, input_spec,
        output_spec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("ReluOpInterface", reluOpQuery, inputShape,
                                     inputLayout, outputShape, outputLayout);
#else
  return std::make_tuple(true, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
AddOpInterface::getOpConstraints(
    const ::llvm::ArrayRef<int64_t> &inputShape_a,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const ::llvm::ArrayRef<int64_t> &inputShape_b,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const ::llvm::ArrayRef<int64_t> &outputShape,
    const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto addOpQuery = [](const ::llvm::ArrayRef<int64_t> &inputShape_a,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
                       const ::llvm::ArrayRef<int64_t> &inputShape_b,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
                       const ::llvm::ArrayRef<int64_t> &outputShape,
                       const ::mlir::tt::ttnn::TTNNLayoutAttr &outputLayout) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::Device *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const ::ttnn::TensorSpec input_spec_a =
        conversion::getTensorSpec(inputShape_a, inputLayout_a);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      input_spec_a.memory_config());
    const ::ttnn::TensorSpec input_spec_b =
        conversion::getTensorSpec(inputShape_b, inputLayout_b);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      input_spec_b.memory_config());
    const ::ttnn::TensorSpec output_spec =
        conversion::getTensorSpec(outputShape, outputLayout);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      output_spec.memory_config());

    return ::ttnn::graph::query_op_constraints(
        ::ttnn::add, device, input_spec_a, input_spec_b,
        output_spec.data_type(),
        output_spec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("AddOpInterface", addOpQuery, inputShape_a,
                                     inputLayout_a, inputShape_b, inputLayout_b,
                                     outputShape, outputLayout);
#else
  return std::make_tuple(true, std::make_tuple(0, 0, 0), std::nullopt);
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
    ::tt::tt_metal::v0::Device *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const ::ttnn::TensorSpec input_spec =
        conversion::getTensorSpec(inputShape, inputLayout);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      input_spec.memory_config());
    const ::ttnn::TensorSpec output_spec =
        conversion::getTensorSpec(outputShape, outputLayout);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      output_spec.memory_config());

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::softmax, device, input_spec, dim_arg,
        output_spec.tensor_layout().get_memory_config());
  };

  return operation::getOpConstraints("SoftmaxOpInterface", softmaxOpQuery,
                                     inputShape, inputLayout, dim_arg,
                                     outputShape, outputLayout);
#else
  return std::make_tuple(true, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
           std::optional<std::string>>
MatmulOpInterface::getOpConstraints(
    const llvm::ArrayRef<int64_t> &inputShape_a,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
    const llvm::ArrayRef<int64_t> &inputShape_b,
    const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
    const llvm::ArrayRef<int64_t> &outputShape,
    const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout, bool transpose_a,
    bool transpose_b) {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto matmulOpQuery = [](const llvm::ArrayRef<int64_t> &inputShape_a,
                          const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_a,
                          const llvm::ArrayRef<int64_t> &inputShape_b,
                          const mlir::tt::ttnn::TTNNLayoutAttr &inputLayout_b,
                          const llvm::ArrayRef<int64_t> &outputShape,
                          const mlir::tt::ttnn::TTNNLayoutAttr &outputLayout,
                          bool transpose_a, bool transpose_b) {
    // open device device, will close it at the end of function
    ::tt::tt_metal::v0::Device *device =
        SingletonDeviceContext::getInstance().getDevice();

    // prepare io specs
    const ::ttnn::TensorSpec input_spec_a =
        conversion::getTensorSpec(inputShape_a, inputLayout_a);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      input_spec_a.memory_config());
    const ::ttnn::TensorSpec input_spec_b =
        conversion::getTensorSpec(inputShape_b, inputLayout_b);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      input_spec_b.memory_config());
    const ::ttnn::TensorSpec output_spec =
        conversion::getTensorSpec(outputShape, outputLayout);
    detail::checkGrid(device->compute_with_storage_grid_size(),
                      output_spec.memory_config());

    // run op constraint query
    return ::ttnn::graph::query_op_constraints(
        ::ttnn::matmul, device, input_spec_a, input_spec_b, transpose_a,
        transpose_b, output_spec.tensor_layout().get_memory_config(),
        output_spec.data_type());
  };

  return operation::getOpConstraints("MatmulOpInterface", matmulOpQuery,
                                     inputShape_a, inputLayout_a, inputShape_b,
                                     inputLayout_b, outputShape, outputLayout,
                                     transpose_a, transpose_b);
#else
  return std::make_tuple(true, std::make_tuple(0, 0, 0), std::nullopt);
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::op_model::ttnn
