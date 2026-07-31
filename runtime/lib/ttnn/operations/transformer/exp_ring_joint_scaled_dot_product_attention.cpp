// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/exp_ring_joint_scaled_dot_product_attention.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::ExpRingJointScaledDotProductAttentionOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());

  // tt-metal is called with dummy joint tensors on the self-attention path, so
  // all three may legitimately be absent. They are all-or-none by construction
  // (enforced by the op verifier).
  auto optionalTensor = [&](const ::tt::target::ttnn::TensorRef *ref)
      -> std::optional<::ttnn::Tensor> {
    return ref ? std::make_optional(tensorPool.getTTNNTensorAndValidate(ref))
               : std::nullopt;
  };
  std::optional<::ttnn::Tensor> jointQuery = optionalTensor(op->joint_query());
  std::optional<::ttnn::Tensor> jointKey = optionalTensor(op->joint_key());
  std::optional<::ttnn::Tensor> jointValue = optionalTensor(op->joint_value());

  // This op runs in tt-metal persistent mode only: the ring scratch buffers and
  // the semaphore pair are materialized in the function prelude and passed in.
  LOG_ASSERT(op->persistent_output_buffer_k() &&
                 op->persistent_output_buffer_v() &&
                 op->multi_device_global_semaphore() &&
                 op->multi_device_global_semaphore()->size() >= 2,
             "exp_ring_joint_scaled_dot_product_attention requires the "
             "persistent K/V buffers and a ping-pong semaphore pool of at "
             "least 2 to be bound");

  ::ttnn::Tensor persistentOutputBufferK =
      tensorPool.getTTNNTensorAndValidate(op->persistent_output_buffer_k());
  ::ttnn::Tensor persistentOutputBufferV =
      tensorPool.getTTNNTensorAndValidate(op->persistent_output_buffer_v());

  std::vector<::ttnn::GlobalSemaphore> multiDeviceGlobalSemaphore;
  multiDeviceGlobalSemaphore.reserve(
      op->multi_device_global_semaphore()->size());
  for (const auto *semaphoreRef : *op->multi_device_global_semaphore()) {
    multiDeviceGlobalSemaphore.push_back(
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            semaphoreRef));
  }

  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;

  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  uint32_t numLinks = op->num_links() ? op->num_links().value() : 1;

  std::optional<float> scale = op->scale();

  // Required, not optional: tt-metal takes the program config by value.
  LOG_ASSERT(op->program_config(),
             "exp_ring_joint_scaled_dot_product_attention requires a program "
             "config");
  ::ttnn::operations::transformer::SDPAProgramConfig programConfig =
      utils::createSDPAProgramConfig(op->program_config());

  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;
  if (op->compute_config()) {
    computeKernelConfig = std::make_optional(
        utils::createDeviceComputeKernelConfig(op->compute_config()));
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  auto [out, jointOut, stats] =
      ::ttnn::transformer::ExecuteExpRingJointAttention::invoke(
          query, key, value, jointQuery, jointKey, jointValue,
          persistentOutputBufferK, persistentOutputBufferV,
          op->joint_strategy()->str(),
          static_cast<std::size_t>(op->logical_n()), programConfig, op->dim(),
          multiDeviceGlobalSemaphore, numLinks, op->cluster_axis(), meshDevice,
          topology, subDeviceId, scale, computeKernelConfig,
          op->num_workers_per_link(), op->num_buffers_per_channel());

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
  tensorPool.insertTTNNTensorAndValidate(op->joint_out(), jointOut);
  tensorPool.insertTTNNTensorAndValidate(op->stats(), stats);
}

} // namespace tt::runtime::ttnn::operations::transformer
