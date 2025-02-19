// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/reduce_scatter.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::ReduceScatterOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  int32_t scatterDimension = op->scatter_dim();
  uint32_t clusterAxis = op->cluster_axis();
  uint32_t numLinks = op->num_links();
  auto reduceType =
      ::tt::runtime::ttnn::utils::getReduceType(op->reduce_type());

  LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::MULTI_DEVICE,
             "Input of reduce_scatter must be MULTIDEVICE. id:",
             op->in()->global_id());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(outputMemoryConfig.has_value(),
             "Memory config must be exist for device tensors");

  ::ttnn::MeshDevice &meshDevice =
      context.getSubMesh(op->device()->global_id());
  ::ttnn::Tensor out = ::ttnn::reduce_scatter(
      input, scatterDimension, clusterAxis, meshDevice, reduceType, numLinks,
      outputMemoryConfig, ::ttnn::ccl::Topology::Linear);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
