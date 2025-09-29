// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

static void runEltwiseBinaryOp(
    const ::tt::target::ttnn::EltwiseBinaryOp *op,
    ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<const ::ttnn::DataType> &,
        const std::optional<::ttnn::MemoryConfig> &,
        std::optional<::ttnn::Tensor>,
        ttsl::Span<const ::ttnn::operations::unary::EltwiseUnaryWithParam>,
        ttsl::Span<const ::ttnn::operations::unary::EltwiseUnaryWithParam>,
        ttsl::Span<const ::ttnn::operations::unary::EltwiseUnaryWithParam>)>
        &ttnnOp) {

  ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
  ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

  std::optional<::ttnn::DataType> outputDataType = std::nullopt;
  if (op->output_dtype()) {
    outputDataType =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig,
                              std::nullopt, {}, {}, {});

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseBinaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseBinaryOpType::Add: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::add);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Multiply: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::multiply);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalRightShift: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_right_shift);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Subtract: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::subtract);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Equal: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::eq);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::NotEqual: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::ne);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterEqual: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::ge);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterThan: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::gt);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessEqual: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::le);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessThan: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::lt);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Divide: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::divide);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalAnd: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalOr: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalXor: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_xor);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
