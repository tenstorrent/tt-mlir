// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary_composite.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const ::ttnn::Tensor &,
                       const std::optional<::ttnn::MemoryConfig> &)> &ttnnOp) {

  ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
  ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<const ::ttnn::DataType> &,
        const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::Tensor> &,
        ttsl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
        ttsl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
        ttsl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
        std::optional<bool>)> &ttnnOp) {

  ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
  ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, std::nullopt, outputMemoryConfig,
                              std::nullopt, {}, {}, {}, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Maximum: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::maximum);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Minimum: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::minimum);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::LogicalLeftShift: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::logical_left_shift);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Remainder: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::remainder);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Pow: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::pow);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Atan2: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::atan2);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseAnd: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::bitwise_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseOr: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::bitwise_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseXor: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::bitwise_xor);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
