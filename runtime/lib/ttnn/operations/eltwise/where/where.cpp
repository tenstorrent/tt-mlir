// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::where {
static void runReductionOp(
    ::tt::target::ttnn::WhereOp const *op, ProgramTensorPool &tensorPool,
    std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &,
        const std::optional<std::variant<int, std::vector<int>>> &, const bool,
        const std::optional<::tt::tt_metal::MemoryConfig> &,
        const std::optional<::ttnn::DeviceComputeKernelConfig> &, float)>
        ttnnOp) {
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());

  ::ttnn::Tensor out = ttnnOp(in, op->pred(), op->on_true(), op->on_false(),
                              outputMemoryConfig /* memory_config_arg */);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::WhereOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  runReductionOp(op, tensorPool, ::ttnn::where);
}
} // namespace tt::runtime::ttnn::operations::where
