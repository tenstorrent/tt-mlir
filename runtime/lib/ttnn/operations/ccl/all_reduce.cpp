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

  // NOTE: The caller is currently responsible for creating/passing semaphores.
  // TODO(hkwon): Remove semaphore creation here once all_reduce_async manages
  // semaphores internally. Tracking:
  // https://github.com/tenstorrent/tt-metal/issues/26952

  // all_reduce_async requires specific numbers of semaphores for different
  // phases:
  // - 2 barrier semaphores for synchronization
  // - 3 reduce-scatter semaphores for the reduce-scatter phase
  // - 2 all-gather semaphores for the all-gather phase
  auto barrier_semaphores = ttnn::utils::createGlobalSemaphores(meshDevice, 2);
  auto rs_semaphores = ttnn::utils::createGlobalSemaphores(meshDevice, 3);
  auto ag_semaphores = ttnn::utils::createGlobalSemaphores(meshDevice, 2);

  ::ttnn::Tensor out = ::ttnn::experimental::all_reduce_async(
      input, clusterAxis, meshDevice, barrier_semaphores, rs_semaphores,
      ag_semaphores, reduceType, outputMemoryConfig,
      ::ttnn::ccl::Topology::Linear,
      std::make_optional(static_cast<size_t>(numLinks)),
      std::nullopt /*worker_subdevice_id_opt*/);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
