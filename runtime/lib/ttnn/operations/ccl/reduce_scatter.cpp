// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/reduce_scatter.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::ReduceScatterOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  int32_t scatterDimension = op->scatter_dim();
  std::optional<uint32_t> clusterAxis =
      std::make_optional<uint32_t>(op->cluster_axis());
  //   auto reduceType =
  //       ::tt::runtime::ttnn::utils::getReduceType(op->reduce_type());
  // TODO(hkwon): Enable reduce_type again once the issue is resolved.
  // Currently the reduce_type argument is commented out because
  // reduce_scatter does not accept it.

  LOG_ASSERT(
      input.storage_type() == ::ttnn::StorageType::DEVICE,
      "Input of reduce_scatter must be DEVICE. id:", op->in()->global_id());
  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  std::optional<::ttnn::Tensor> optionalOutputTensor = std::nullopt;
  // Do not pass optionalOutputTensor to reduce_scatter for now.
  // Enable it when needed.

  std::optional<uint32_t> numLinks = op->num_links();
  std::optional<::tt::tt_fabric::Topology> topology = std::nullopt;
  if (op->topology()) {
    topology = std::make_optional<::tt::tt_fabric::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  ::ttnn::Tensor out = ::ttnn::reduce_scatter(
      input, scatterDimension, clusterAxis, subDeviceId, outputMemoryConfig,
      /*intermediate_memory_config=*/std::nullopt, optionalOutputTensor,
      numLinks, topology,
      /*chunks_per_sync=*/std::nullopt,
      /*num_workers_per_link=*/std::nullopt,
      /*num_buffers_per_channel=*/std::nullopt,
      /*compute_kernel_config=*/computeConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
