// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "unary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

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

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Abs: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::abs);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Cbrt: {
    runEltwiseUnaryCompositeOP(op, tensorPool, ::ttnn::cbrt);
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
  case ::tt::target::ttnn::EltwiseOpType::Reciprocal: {
    runEltwiseUnaryOP(op, tensorPool, ::ttnn::reciprocal);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::exp);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Ceil: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::ceil);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Pow: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::pow);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sine: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::sine);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Cosine: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool, ::ttnn::cosine);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Logit: {
    runEltwiseUnaryWithFastAndApproximateModeOP(op, tensorPool,
                                                ::ttnn::logistic);
    break;
  }
  default:
    throw std::invalid_argument("Unsupported unary operation");
  }
}

} // namespace tt::runtime::ttnn::operations::unary
