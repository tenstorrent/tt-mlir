// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "unary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttnn/operations/copy.hpp"

namespace tt::runtime::ttnn::operations::unary {

static void
getEltwiseUnaryOPInputTensor(const ::tt::target::ttnn::EltwiseOp *op,
                             ProgramTensorPool &tensorPool,
                             ::ttnn::Tensor **in) {
  LOG_ASSERT(op->ins()->size() == 1, "Expected 1 input, got ",
             op->ins()->size());
  *in = &(tensorPool.at(op->ins()->Get(0)->global_id()));
  DEBUG_ASSERT((*in)->is_allocated());
}

static void runEltwiseUnaryOP(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &,
                       const std::optional<::tt::tt_metal::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)>
        ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOPInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*in, outputMemoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

static void runEltwiseUnaryWithFastAndApproximateModeOP(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const bool,
                       const std::optional<::tt::tt_metal::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)>
        ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOPInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out =
      ttnnOp(*in, false /* parameter */, outputMemoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

static void runTypecastOp(const ::tt::target::ttnn::EltwiseOp *op,
                          ProgramTensorPool &tensorPool) {
  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOPInputTensor(op, tensorPool, &in);
  const DataType outputType = tensorPool.at(op->out()->global_id()).get_dtype();
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ::ttnn::operations::copy::Typecast().invoke(
      *in, outputType, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Abs: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::abs);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalNot: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::logical_not);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Neg: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::neg);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Relu: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::relu);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sqrt: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::sqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Rsqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::rsqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sigmoid: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::sigmoid);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Typecast: {
    runTypecastOp(op, tensorPool);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Reciprocal: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::reciprocal);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::exp);
    break;
  }
  default:
    throw std::invalid_argument("Unsupported unary operation");
  }
}

} // namespace tt::runtime::ttnn::operations::unary
