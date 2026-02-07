// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/mesh_partition.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include <optional>
namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::MeshPartitionOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  LOG_ASSERT(
      input.storage_type() == ::ttnn::StorageType::DEVICE,
      "Input of mesh_partition must be DEVICE. id:", op->in()->global_id());
  int32_t dim = op->dim();
  std::optional<uint32_t> clusterAxis = op->cluster_axis();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  ::ttnn::Tensor out =
      ::ttnn::mesh_partition(input, dim, clusterAxis, outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
