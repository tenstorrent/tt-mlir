// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/ternary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::ternary {

static void runEltwiseTernaryWhereOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<::tt::tt_metal::MemoryConfig> &)> &ttnnOp) {
  ::ttnn::Tensor *first = nullptr;
  ::ttnn::Tensor *second = nullptr;
  ::ttnn::Tensor *third = nullptr;
  getEltwiseTernaryOpInputTensors(op, tensorPool, &first, &second, &third);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*first, *second, *third, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Where: {
    runEltwiseTernaryWhereOp(op, tensorPool, ::ttnn::where);
    break;
  }
  default:
    LOG_FATAL("Unsupported ternary operation");
  }
}
} // namespace tt::runtime::ttnn::operations::ternary
