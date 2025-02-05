// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_gather.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllGatherOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  int32_t allGatherDim = op->all_gather_dim();
  uint32_t clusterAxis = op->cluster_axis();
  uint32_t numLinks = op->num_links();
  LOG_ASSERT(
      input.storage_type() == ::tt::tt_metal::StorageType::MULTI_DEVICE,
      "Input of all_gather must be MULTIDEVICE. id:", op->in()->global_id());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  ::ttnn::MeshDevice &meshDevice =
      context.getSubMesh(op->device()->global_id());
  ::ttnn::Tensor out =
      ::ttnn::all_gather(input, allGatherDim, clusterAxis, meshDevice, numLinks,
                         outputMemoryConfig, std::nullopt, std::nullopt,
                         ::ttnn::ccl::Topology::Linear);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
