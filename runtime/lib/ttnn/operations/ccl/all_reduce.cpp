// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_reduce.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllReduceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  uint32_t clusterAxis = op->cluster_axis();
  uint32_t numLinks = op->num_links();
  auto reduceType =
      ::tt::runtime::ttnn::utils::getReduceType(op->reduce_type());
  LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::DEVICE,
             "Input of all_reduce must be DEVICE. id:", op->in()->global_id());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::Tensor out;
  auto createGlobalSemaphore = [&meshDevice]() {
    return ::ttnn::global_semaphore::create_global_semaphore(
        &meshDevice,
        meshDevice.worker_cores(::tt::tt_metal::HalProgrammableCoreType::TENSIX,
                                tt::tt_metal::SubDeviceId{0}),
        0);
  };
  std::vector<::ttnn::GlobalSemaphore> barrier_semaphores;
  std::generate_n(std::back_inserter(barrier_semaphores), 2,
                  createGlobalSemaphore);
  std::vector<::ttnn::GlobalSemaphore> rs_global_semaphores;
  std::generate_n(std::back_inserter(rs_global_semaphores), 3,
                  createGlobalSemaphore);
  std::vector<::ttnn::GlobalSemaphore> ag_global_semaphores;
  std::generate_n(std::back_inserter(ag_global_semaphores), 2,
                  createGlobalSemaphore);
  out = ::ttnn::experimental::all_reduce_async(
      input, clusterAxis, meshDevice, barrier_semaphores, rs_global_semaphores,
      ag_global_semaphores, reduceType, outputMemoryConfig,
      ::ttnn::ccl::Topology::Linear,
      std::make_optional(static_cast<size_t>(numLinks)), std::nullopt);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
