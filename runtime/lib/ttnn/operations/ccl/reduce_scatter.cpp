// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/reduce_scatter.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_async/reduce_scatter.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::ReduceScatterOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  int32_t scatterDimension = op->scatter_dim();
  uint32_t clusterAxis = op->cluster_axis();
  uint32_t numLinks = op->num_links();
  auto reduceType =
      ::tt::runtime::ttnn::utils::getReduceType(op->reduce_type());

  LOG_ASSERT(
      input.storage_type() == ::ttnn::StorageType::DEVICE,
      "Input of reduce_scatter must be DEVICE. id:", op->in()->global_id());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  ::ttnn::Tensor out;
  if (!context.getDeviceHandle().isFabricEnabled()) {
    out = ::ttnn::reduce_scatter(
        input, scatterDimension, clusterAxis, meshDevice, reduceType, numLinks,
        outputMemoryConfig, ::ttnn::ccl::Topology::Linear);
  } else {
    std::vector<::ttnn::GlobalSemaphore> semaphores;
    auto from_semaphore = ::ttnn::global_semaphore::create_global_semaphore(
        &meshDevice,
        meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                                tt::tt_metal::SubDeviceId{0}),
        0, tt::tt_metal::BufferType::L1);
    auto to_semaphore = ::ttnn::global_semaphore::create_global_semaphore(
        &meshDevice,
        meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                                tt::tt_metal::SubDeviceId{0}),
        0, tt::tt_metal::BufferType::L1);
    out = ::ttnn::experimental::reduce_scatter_async(
        input, scatterDimension, clusterAxis, meshDevice, from_semaphore,
        to_semaphore, std::nullopt, reduceType, outputMemoryConfig,
        ::ttnn::ccl::Topology::Linear,
        std::make_optional(static_cast<size_t>(numLinks)), std::nullopt);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
