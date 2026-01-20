// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_reduce.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllReduceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  uint32_t clusterAxis = op->cluster_axis();
  LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::DEVICE,
             "Input of all_reduce must be DEVICE. id:", op->in()->global_id());
  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId =
      op->sub_device_id() ? std::make_optional<::tt::tt_metal::SubDeviceId>(
                                op->sub_device_id().value())
                          : std::nullopt;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<uint32_t> numLinks = op->num_links();
  std::optional<::tt::tt_fabric::Topology> topology = ::tt::tt_fabric::Topology::Ring;
  std::cout << "AllReduce Topology: "
            << static_cast<int>(topology.value()) << std::endl;
//  std::optional<::tt::tt_fabric::Topology> topology = std::nullopt;
//  if (op->topology()) {
//    topology = std::make_optional<::tt::tt_fabric::Topology>(
//        ::tt::runtime::common::toMetalTopology(op->topology().value()));
//  }

  ::ttnn::Tensor out = ::ttnn::all_reduce(
      input, clusterAxis, subDeviceId, outputMemoryConfig, numLinks, topology);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
