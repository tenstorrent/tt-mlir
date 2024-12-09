// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "unary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/unary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/copy.hpp"

namespace tt::runtime::ttnn::operations::unary {

static void runEltwiseUnaryOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &,
                       const std::optional<::tt::tt_metal::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)> &ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOpInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*in, outputMemoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

static void runEltwiseUnaryWithFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const bool,
                       const std::optional<::tt::tt_metal::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)> &ttnnOp) {

  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOpInputTensor(op, tensorPool, &in);

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out =
      ttnnOp(*in, false /* parameter */, outputMemoryConfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

static void runEltwiseUnaryWithFloatParameterOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(const ::ttnn::Tensor &, float,
                                       const ::tt::tt_metal::MemoryConfig &)>
        &ttnnOp) {
  ::ttnn::Tensor *in = nullptr;
  getEltwiseUnaryOpInputTensor(op, tensorPool, &in);

  float parameter = op->params_as_EltwiseOpWithFloatParams()->parameter();
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  ::ttnn::Tensor out = ttnnOp(*in, parameter, outputMemoryConfig);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Abs: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::abs);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Ceil: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::ceil);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Cos: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::cos);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Floor: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::floor);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Gelu: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::gelu);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::IsFinite: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::isfinite);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalNot: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::logical_not);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Neg: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::neg);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Relu: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::relu);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sqrt: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Rsqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::rsqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sigmoid: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sigmoid);
    break;
  }

  case ::tt::target::ttnn::EltwiseOpType::Sin: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sin);
    break;
  }

  case ::tt::target::ttnn::EltwiseOpType::Reciprocal: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::reciprocal);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Sign: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sign);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Tan: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::tan);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Tanh: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::tanh);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::exp);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Log: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::log);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Expm1: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::expm1);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LeakyRelu: {
    runEltwiseUnaryWithFloatParameterOp(op, tensorPool, ::ttnn::leaky_relu);
    break;
  }
  default:
    LOG_FATAL("Unsupported unary operation");
  }
}

} // namespace tt::runtime::ttnn::operations::unary
