// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/reduce_scatter.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"

// NOTE: Temporarily using reduce_scatter_minimal_async due to an issue in
// tt-metal. https://github.com/tenstorrent/tt-metal/issues/25212

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::ReduceScatterOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  int32_t scatterDimension = op->scatter_dim();
  uint32_t clusterAxis = op->cluster_axis();
  uint32_t numLinks = op->num_links();
  //   auto reduceType =
  //       ::tt::runtime::ttnn::utils::getReduceType(op->reduce_type());
  // TODO(hkwon): Enable reduce_type again once the issue is resolved.
  // Currently the reduce_type argument is commented out because
  // reduce_scatter_minimal_async does not accept it.

  LOG_ASSERT(
      input.storage_type() == ::ttnn::StorageType::DEVICE,
      "Input of reduce_scatter must be DEVICE. id:", op->in()->global_id());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  ::ttnn::Shape outputShape =
      operations::utils::toTTNNShape(*op->out()->desc()->shape());
  // Line RS requires double-sized input for forward/backward
  const auto &originalShape = *op->in()->desc()->shape();
  std::vector<uint32_t> modifiedShape{2}; // Start with '2'
  modifiedShape.insert(modifiedShape.end(), originalShape.begin(),
                       originalShape.end()); // Fill the rest of the shape
  ::ttnn::Shape intermediateShape = ::ttnn::Shape(modifiedShape);
  ::ttnn::DataType ttnnDtype = ::tt::runtime::ttnn::utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  ::ttnn::Layout ttnnLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());

  ::ttnn::Tensor intermediateBuffer =
      ::ttnn::empty(intermediateShape, ttnnDtype, ttnnLayout, &meshDevice,
                    outputMemoryConfig.value());
  ::ttnn::Tensor outputBuffer =
      ::ttnn::empty(outputShape, ttnnDtype, ttnnLayout, &meshDevice,
                    outputMemoryConfig.value());
  std::vector<::ttnn::Tensor> persistent_buffers = {intermediateBuffer,
                                                    outputBuffer};

  std::vector<::ttnn::GlobalSemaphore> semaphores;
  // reduce_scatter_minimal_async requires 3 semaphores.
  for (int i = 0; i < 3; i++) {
    semaphores.push_back(::ttnn::global_semaphore::create_global_semaphore(
        &meshDevice,
        meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                                tt::tt_metal::SubDeviceId{0}),
        0, tt::tt_metal::BufferType::L1));
  }
  ::ttnn::Tensor out = ::ttnn::experimental::reduce_scatter_minimal_async(
      input, std::make_optional(persistent_buffers), scatterDimension,
      semaphores, std::nullopt, numLinks, outputMemoryConfig.value(),
      outputMemoryConfig.value(), ::ttnn::ccl::Topology::Linear, std::nullopt,
      clusterAxis);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
