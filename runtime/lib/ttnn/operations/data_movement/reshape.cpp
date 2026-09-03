// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  // Mainline reshape is only safe on Quasar when it degenerates to a view. A
  // tiled reshape goes through reshape_tiled -> prim::reshape_view, whose
  // program factory builds a DataMovementKernel and TT_FATALs. Measured: the
  // reshape in the global-avg-pool and max-pool graphs takes exactly that path.
  ::ttnn::Tensor out =
      utils::isQuasar()
          ? ::ttnn::operations::experimental::quasar::reshape(in, shape,
                                                             memoryConfig)
          : ::ttnn::reshape(in, shape, memoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
