// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/distributed_layer_norm.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/dit_fused_distributed_rmsnorm.hpp"

namespace tt::runtime::ttnn::operations::distributed_layer_norm {
void run(const ::tt::target::ttnn::DistributedLayerNormOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  uint32_t clusterAxis = op->cluster_axis();
  float epsilon = op->epsilon();

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<size_t> numLinks = std::nullopt;
  if (op->num_links()) {
    numLinks = static_cast<size_t>(op->num_links().value());
  }

  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Ring;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  LOG_ASSERT(op->semaphore(),
             "DistributedLayerNormOp expects an explicit global semaphore");
  ::ttnn::GlobalSemaphore semaphore =
      context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
          op->semaphore());
  std::vector<::ttnn::GlobalSemaphore> semaphores{semaphore};

  std::optional<::ttnn::Tensor> stats = std::nullopt;
  if (op->stats()) {
    stats = tensorPool.getTTNNTensorAndValidate(op->stats());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  ::ttnn::Tensor output = ::ttnn::experimental::dit_fused_distributed_layernorm(
      input, clusterAxis, meshDevice, semaphores, topology, epsilon,
      /*num_heads_per_device=*/1, weight, bias,
      /*transformation_mat=*/std::nullopt, /*rope_cos=*/std::nullopt,
      /*rope_sin=*/std::nullopt, /*dtype=*/std::nullopt, stats, numLinks,
      /*subdevice_id=*/std::nullopt, memoryConfig, computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::distributed_layer_norm
