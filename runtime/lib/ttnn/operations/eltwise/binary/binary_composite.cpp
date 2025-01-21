// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary_composite.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/binary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::binary::composite {

static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const ::ttnn::Tensor &,
                       const std::optional<::ttnn::MemoryConfig> &)> &ttnnOp) {

  ::ttnn::Tensor *lhs = nullptr;
  ::ttnn::Tensor *rhs = nullptr;
  getEltwiseBinaryOpInputTensors(op, tensorPool, &lhs, &rhs);

  ::ttnn::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Maximum: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::maximum);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Minimum: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::minimum);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Remainder: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::remainder);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Scatter: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::scatter);
    break;
  }
  default:
    LOG_FATAL("Unsupported Eltwise Binary Composite operation");
  }
}

} // namespace tt::runtime::ttnn::operations::binary::composite
