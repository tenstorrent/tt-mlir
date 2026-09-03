// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/dit_fused_distributed_rmsnorm.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/core.hpp"

namespace tt::runtime::ttnn::operations::dit_fused_distributed_rmsnorm {

namespace {
// Rank-3 activations `[1, N, H]` are `[1, 1, N, H]` to metal. Last-two dims
// already tile, so this is a view. Do not rank-up TILE γ here: torch `[H]`
// is reshaped to `[1, H]` in TTIR before tilize.
::ttnn::Tensor unsqueezeInputToMetalLayout(const ::ttnn::Tensor &tensor) {
  if (tensor.logical_shape().rank() >= 4) {
    return tensor;
  }
  return ::ttnn::unsqueeze_to_4D(tensor);
}
} // namespace

void run(const ::tt::target::ttnn::DitFusedDistributedRmsnormOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor input = unsqueezeInputToMetalLayout(
      tensorPool.getTTNNTensorAndValidate(op->input()));

  std::optional<::ttnn::Tensor> weight = std::nullopt;
  if (op->weight()) {
    weight = tensorPool.getTTNNTensorAndValidate(op->weight());
  }

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<::ttnn::Tensor> transformationMat = std::nullopt;
  if (op->transformation_mat()) {
    transformationMat =
        tensorPool.getTTNNTensorAndValidate(op->transformation_mat());
  }

  std::optional<::ttnn::Tensor> ropeCos = std::nullopt;
  if (op->rope_cos()) {
    ropeCos = tensorPool.getTTNNTensorAndValidate(op->rope_cos());
  }

  std::optional<::ttnn::Tensor> ropeSin = std::nullopt;
  if (op->rope_sin()) {
    ropeSin = tensorPool.getTTNNTensorAndValidate(op->rope_sin());
  }

  uint32_t clusterAxis = op->cluster_axis();
  float epsilon = op->epsilon();
  uint32_t numHeadsPerDevice = op->num_heads_per_device();
  bool perHeadNorm = op->per_head_norm();

  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;

  std::optional<size_t> numLinks =
      static_cast<size_t>(op->num_links() ? op->num_links().value() : 1u);

  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Ring;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  std::vector<::ttnn::GlobalSemaphore> semaphores;
  if (op->semaphore()) {
    semaphores.push_back(
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            op->semaphore()));
  }

  uint32_t numLinksForStats = numLinks.value_or(1u);
  // Always take metal's stats buffer. An IR-allocated EmptyOp can disagree
  // with `compute_sizing` (fabric payload, compute grid) and either TT_FATAL
  // or silently corrupt the all-gather.
  std::optional<::ttnn::Tensor> stats =
      ::ttnn::experimental::dit_fused_distributed_rmsnorm_create_stats_buffer(
          input, clusterAxis, meshDevice, numHeadsPerDevice, perHeadNorm,
          numLinksForStats, weight, transformationMat, ropeCos, ropeSin);

  ::ttnn::Tensor output = ::ttnn::experimental::dit_fused_distributed_rmsnorm(
      input, clusterAxis, meshDevice, semaphores, topology, epsilon,
      numHeadsPerDevice, perHeadNorm, weight, bias, transformationMat, ropeCos,
      ropeSin, dtype, stats, numLinks, subDeviceId, memoryConfig,
      computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::dit_fused_distributed_rmsnorm
