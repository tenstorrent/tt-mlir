// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/unary/unary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace tt::runtime::ttnn::operations::eltwise::unary {

static void runEltwiseUnaryOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::Tensor> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(in, outputMemoryConfig, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryTanhOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::Tensor> &, bool)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ttnnOp(in, outputMemoryConfig, std::nullopt, /* accuracy= */ true);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryWithFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const bool,
                       const std::optional<::ttnn::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ttnnOp(in, /*parameter=*/false, outputMemoryConfig, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryWithVectorAndFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const int, const bool,
                       const std::optional<::ttnn::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ttnnOp(in, static_cast<int>(::ttnn::operations::unary::VecMode::RC),
             /*parameter=*/false, outputMemoryConfig, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryWithFloatParameterOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, float,
                       const std::optional<::ttnn::MemoryConfig> &)> &ttnnOp) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  float parameter = op->params_as_EltwiseOpWithFloatParams()->parameter();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(in, parameter, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseUnaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseUnaryOpType::Abs: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::abs);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Ceil: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::ceil);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Cos: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::cos);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Floor: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::floor);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Gelu: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::gelu);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::IsFinite: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::isfinite);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LogicalNot: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::logical_not);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Neg: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::neg);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Relu: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::relu);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sqrt: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Rsqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::rsqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid: {
    runEltwiseUnaryWithVectorAndFastAndApproximateModeOp(op, tensorPool,
                                                         ::ttnn::sigmoid);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sin: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sin);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Reciprocal: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::reciprocal);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sign: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sign);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tan: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::tan);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tanh: {
    runEltwiseUnaryTanhOp(op, tensorPool, ::ttnn::tanh);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Atan: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::atan);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::exp);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Log: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::log);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Expm1: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::expm1);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu: {
    runEltwiseUnaryWithFloatParameterOp(op, tensorPool, ::ttnn::leaky_relu);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::BitwiseNot: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::bitwise_not);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Erf: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::erf);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Erfc: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::erfc);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary
