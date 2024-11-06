// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "binary_composite.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/binary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"

namespace tt::runtime::ttnn::operations::binary::composite {

static void runEltwiseBinaryCompositeOP(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const ::ttnn::Tensor &,
                       const std::optional<::tt::tt_metal::MemoryConfig> &)>
        ttnnOp) {

  ::ttnn::Tensor *lhs = nullptr;
  ::ttnn::Tensor *rhs = nullptr;
  getEltwiseBinaryOPInputTensors(op, tensorPool, &lhs, &rhs);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Maximum: {
    runEltwiseBinaryCompositeOP(op, tensorPool, ::ttnn::maximum);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Minimum: {
    runEltwiseBinaryCompositeOP(op, tensorPool, ::ttnn::minimum);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Remainder: {
    runEltwiseBinaryCompositeOP(op, tensorPool, ::ttnn::remainder);
    break;
  }
  default:
    throw std::invalid_argument(
        "Unsupported Eltwise Binary Composite operation");
  }
}

} // namespace tt::runtime::ttnn::operations::binary::composite
