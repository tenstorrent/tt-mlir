// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/ternary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::ternary {

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  if (op->type() != ::tt::target::ttnn::EltwiseOpType::Where) {
    throw std::invalid_argument("Unsupported Eltwise Ternary operation");
  }

  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor *first = nullptr;
  ::ttnn::Tensor *second = nullptr;
  ::ttnn::Tensor *third = nullptr;
  getEltwiseTernaryOPInputTensors(op, tensorPool, &first, &second, &third);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out =
      ::ttnn::where(*first, *second, *third, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::ternary
