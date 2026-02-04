// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/distributed_rms_norm.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace tt::runtime::ttnn::operations::distributed_rms_norm {
void run(const ::tt::target::ttnn::DistributedRMSNormOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  // Handle optional weight parameter
  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  // Handle optional residual parameter
  std::optional<::ttnn::Tensor> residual = std::nullopt;
  if (op->residual()) {
    residual = tensorPool.getTTNNTensorAndValidate(op->residual());
  }

  uint32_t clusterAxis = op->cluster_axis();
  float epsilon = op->epsilon();

  // Handle optional sub_device_id
  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;

  // Handle optional memory config
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  // Handle optional num_links
  std::optional<size_t> numLinks = std::nullopt;
  if (op->num_links()) {
    numLinks = static_cast<size_t>(op->num_links().value());
  }

  // Handle optional topology
  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  // Handle optional compute config
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  // Get MeshDevice from context
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  // Create program config from input's shard spec
  auto shardSpec = input.shard_spec();
  ::ttnn::prim::LayerNormProgramConfig programConfig =
      ::ttnn::prim::create_program_config(shardSpec);

  // Create GlobalSemaphore from shard spec grid
  // The semaphore is created internally at runtime (like all_gather does)
  LOG_ASSERT(shardSpec.has_value(),
             "Input tensor must have shard spec for distributed_rms_norm");
  auto semaphore = ::ttnn::global_semaphore::create_global_semaphore(
      &meshDevice, shardSpec->grid, 0);

  // Call fused_rms_minimal
  ::ttnn::Tensor output = ::ttnn::fused_rms_minimal(
      input, programConfig, clusterAxis, meshDevice, semaphore,
      /*persistent_output_tensor=*/std::nullopt, numLinks, topology,
      subDeviceId,
      /*dtype=*/std::nullopt, computeConfig, memoryConfig, residual, epsilon,
      weight,
      /*stats=*/std::nullopt,
      /*use_noc1_only=*/false);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::distributed_rms_norm
