// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "unary_composite.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/unary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace tt::runtime::ttnn::operations::unary::composite {

static void runEltwiseUnaryCompositeOP(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    std::function<::ttnn::Tensor(const ::ttnn::Tensor &,
                                 const ::tt::tt_metal::MemoryConfig &)>
        ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOPInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*in, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Cbrt: {
    runEltwiseUnaryCompositeOP(op, tensorPool, ::ttnn::cbrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Log1p: {
    runEltwiseUnaryCompositeOP(op, tensorPool, ::ttnn::log1p);
    break;
  }
  default:
    throw std::invalid_argument(
        "Unsupported Eltwise Binary Composite operation");
  }
}

} // namespace tt::runtime::ttnn::operations::unary::composite
