// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/unary/unary_composite.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace tt::runtime::ttnn::operations::eltwise::unary {

static void runEltwiseUnaryCompositeOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const std::optional<::ttnn::MemoryConfig> &)>
        &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(in, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryCompositeClampScalarOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  float min = op->params_as_ClampScalarOpParams()->min();
  float max = op->params_as_ClampScalarOpParams()->max();

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::clamp(in, min, max, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryCompositeClampTensorOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttnn::Tensor min = tensorPool.getTTNNTensorAndValidate(
      op->params_as_ClampTensorOpParams()->min());
  ::ttnn::Tensor max = tensorPool.getTTNNTensorAndValidate(
      op->params_as_ClampTensorOpParams()->max());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::clamp(in, min, max, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryCompositeWithFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const bool,
                       const std::optional<::ttnn::MemoryConfig> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ttnnOp(in, /*fast_and_approximate_mode=*/false, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Cbrt: {
    runEltwiseUnaryCompositeOp(op, tensorPool, ::ttnn::cbrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar: {
    runEltwiseUnaryCompositeClampScalarOp(op, tensorPool);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor: {
    runEltwiseUnaryCompositeClampTensorOp(op, tensorPool);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Log1p: {
    runEltwiseUnaryCompositeWithFastAndApproximateModeOp(op, tensorPool,
                                                         ::ttnn::log1p);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary
