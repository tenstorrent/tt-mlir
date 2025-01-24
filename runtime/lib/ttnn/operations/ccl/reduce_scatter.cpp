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
  int32_t scatterSplitDim = op->scatter_split_dim();
  int32_t numLinks = op->num_links();
  auto mathOp =
      static_cast<::ttnn::operations::reduction::ReduceType>(op->math_op());
  // Reduction in horizontal direction (x-dimension) in linear computation
  // config: e.g., For 2x4 mesh, clusterAxis (1) means reduction in horizontal
  // direction such as 0,1,2,3 and 4,5,6,7.
  int32_t clusterAxis = 1;
  LOG_ASSERT(input.storage_type() == ::tt::tt_metal::StorageType::MULTI_DEVICE,
             "Input of reduce_scatter must be MULTIDEVICE. id:",
             op->in()->global_id());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  ::ttnn::MeshDevice &meshDevice =
      context.getSubMesh(op->device()->global_id());
  ::ttnn::Tensor out = ::ttnn::reduce_scatter(
      input, scatterSplitDim, clusterAxis, meshDevice, mathOp, numLinks,
      outputMemoryConfig, ::ttnn::ccl::Topology::Linear);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
