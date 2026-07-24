// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/minimal_matmul_strided_reduce_scatter_async.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/minimal_matmul_strided_reduce_scatter_async.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::MinimalMatmulStridedReduceScatterAsyncOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<::ttnn::Tensor> addcmulInput1 = std::nullopt;
  if (op->addcmul_input1()) {
    addcmulInput1 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input1());
  }

  std::optional<::ttnn::Tensor> addcmulInput2 = std::nullopt;
  if (op->addcmul_input2()) {
    addcmulInput2 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input2());
  }

  std::optional<float> scalar = std::nullopt;
  if (op->scalar()) {
    scalar = op->scalar().value();
  }

  // The async reduce-scatter is synchronized by the multi-device global
  // semaphores plus an optional barrier semaphore.
  std::vector<::ttnn::GlobalSemaphore> multiDeviceSemaphore;
  for (const auto *semaphoreRef : *op->multi_device_semaphore()) {
    multiDeviceSemaphore.push_back(
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            semaphoreRef));
  }

  std::optional<::ttnn::GlobalSemaphore> barrierSemaphore = std::nullopt;
  if (op->barrier_semaphore()) {
    barrierSemaphore =
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            op->barrier_semaphore());
  }

  // Topology is a required metal parameter; default to Ring when unset.
  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Ring;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*op->dtype());
  }

  uint32_t numLinks = op->num_links() ? op->num_links().value() : 1;

  std::optional<uint32_t> clusterAxis = std::nullopt;
  if (op->cluster_axis()) {
    clusterAxis = op->cluster_axis().value();
  }

  // compute_kernel_config is a required tt-metal parameter. Build one from the
  // op attribute when present, otherwise fall back to the device-default.
  ::ttnn::DeviceComputeKernelConfig computeKernelConfig =
      op->compute_config()
          ? utils::createDeviceComputeKernelConfig(op->compute_config())
          : ::ttnn::init_device_compute_kernel_config(
                context.getMeshDevice().arch(), /*device_kernel_config=*/
                std::nullopt);

  // The reduce-scatter core-grid offset, `MinimalMatmulConfig`, fused
  // activation, persistent buffers and FSDP path are not modeled by the
  // compiler yet; pass their tt-metal defaults.
  ::ttnn::CoreCoord reduceScatterCoreGridOffset{0, 0};

  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::experimental::minimal_matmul_strided_reduce_scatter_async(
          input, weight, static_cast<uint32_t>(op->dim()), multiDeviceSemaphore,
          reduceScatterCoreGridOffset, computeKernelConfig, numLinks,
          /*memory_config_mm=*/std::nullopt,
          /*rs_output_mem_config=*/memoryConfig,
          /*rs_intermediate_mem_config=*/std::nullopt, topology, clusterAxis,
          bias, /*fused_activation=*/std::nullopt, /*config=*/std::nullopt,
          barrierSemaphore, /*using_persistent_buffers=*/false,
          /*sub_device_id=*/std::nullopt,
          /*num_workers_per_link=*/
          std::make_optional(op->num_workers_per_link()),
          /*num_buffers_per_channel=*/
          std::make_optional(op->num_buffers_per_channel()),
          /*chunk_width_in_mm_blocks=*/std::nullopt,
          /*optional_rs_output_tensor=*/std::nullopt,
          /*fused_ternary_scalar=*/scalar, addcmulInput1, addcmulInput2, dtype);

  const auto *outputRefs = op->outputs();
  LOG_ASSERT(outputs.size() == outputRefs->size(),
             "minimal_matmul_strided_reduce_scatter_async produced ",
             outputs.size(), " outputs but the flatbuffer expects ",
             outputRefs->size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(outputRefs->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::ccl
