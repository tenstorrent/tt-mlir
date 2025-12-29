// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_gather.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllGatherOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  int32_t allGatherDim = op->all_gather_dim();
  uint32_t clusterAxis = op->cluster_axis();
  LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::DEVICE,
             "Input of all_gather must be DEVICE. id:", op->in()->global_id());
  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  std::optional<::ttnn::Tensor> optionalOutputTensor = std::nullopt;
  // Do not pass optionalOutputTensor to all_gather for now.
  // Enable it when needed.

  std::optional<uint32_t> numLinks = op->num_links();
  //std::optional<::tt::tt_fabric::Topology> topology = std::nullopt;
  //if (op->topology()) {
  //  topology = std::make_optional<::tt::tt_fabric::Topology>(
  //      ::tt::runtime::common::toMetalTopology(op->topology().value()));
  //}

  std::optional<::tt::tt_fabric::Topology> topology = ::tt::tt_fabric::Topology::Ring;
  std::cout << "AllGather Topology: "
            << static_cast<int>(topology.value()) << std::endl;

  ::ttnn::Tensor out = ::ttnn::all_gather(
      input, allGatherDim, clusterAxis, subDeviceId, outputMemoryConfig,
      optionalOutputTensor, numLinks, topology);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
