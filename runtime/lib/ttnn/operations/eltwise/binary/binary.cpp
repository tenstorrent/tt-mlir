// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/binary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace tt::runtime::ttnn::operations::binary {

static void runEltwiseBinaryOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<const ::ttnn::DataType> &,
        const std::optional<::tt::tt_metal::MemoryConfig> &,
        std::optional<::ttnn::Tensor>,
        std::optional<::ttnn::operations::unary::FusedActivations>,
        std::optional<::ttnn::operations::unary::UnaryWithParam>)> &ttnnOp) {

  ::ttnn::Tensor *lhs = nullptr;
  ::ttnn::Tensor *rhs = nullptr;
  getEltwiseBinaryOpInputTensors(op, tensorPool, &lhs, &rhs);

  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig,
                              std::nullopt, std::nullopt, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseOpType::Add: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::add);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::multiply);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Subtract: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::subtract);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Equal: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::eq);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::NotEqual: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::ne);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterEqual: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::ge);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterThan: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::gt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LessEqual: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::le);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LessThan: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::lt);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Div: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::divide);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalAnd: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalOr: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::LogicalXor: {
    runEltwiseBinaryOp(op, tensorPool, ::ttnn::logical_xor);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::BitwiseAnd: {
    LOG_ASSERT(false, "Binary bitwise_and op not supported in ttnn. See "
                      "https://github.com/tenstorrent/tt-metal/issues/13582");
    // runEltwiseBinaryOP(op, tensorPool, ::ttnn::bitwise_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::BitwiseOr: {
    LOG_ASSERT(false, "Binary bitwise_or op not supported in ttnn. See "
                      "https://github.com/tenstorrent/tt-metal/issues/13582");
    // runEltwiseBinaryOP(op, tensorPool, ::ttnn::bitwise_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::BitwiseXor: {
    LOG_ASSERT(false, "Binary bitwise_xor op not supported in ttnn. See "
                      "https://github.com/tenstorrent/tt-metal/issues/13582");
    // runEltwiseBinaryOP(op, tensorPool, ::ttnn::bitwise_xor);
    break;
  }
  default:
    LOG_FATAL("Unsupported Eltwise Binary operation");
  }
}

} // namespace tt::runtime::ttnn::operations::binary
