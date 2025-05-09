// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

// static void runEltwiseBinaryOp(
//     const ::tt::target::ttnn::EltwiseBinaryOp *op,
//     ProgramTensorPool &tensorPool,
//     const std::function<::ttnn::Tensor(
//         const ::ttnn::Tensor &, const ::ttnn::Tensor &,
//         const std::optional<const ::ttnn::DataType> &,
//         const std::optional<::ttnn::MemoryConfig> &,
//         std::optional<::ttnn::Tensor>,
//         tt::stl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
//         tt::stl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
//         tt::stl::Span<const ::ttnn::operations::unary::UnaryWithParam>)>
//         &ttnnOp) {

//   ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
//   ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

//   if (operations::utils::shouldSwapBinaryOperands(*lhs, *rhs)) {
//     std::swap(lhs, rhs);
//   }

//   std::optional<::ttnn::DataType> outputDataType = std::nullopt;
//   if (op->output_dtype()) {
//     outputDataType =
//         ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->output_dtype()));
//   }

//   std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
//       ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
//           op->memory_config());
//   LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
//                  outputMemoryConfig.has_value(),
//              "Memory config must exist for device tensors");

//   ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig,
//                               std::nullopt, {}, {}, {});

//   tensorPool.insertTTNNTensorAndValidate(op->out(), out);
// }

static void runEltwiseNGBinaryOp(
    const ::tt::target::ttnn::EltwiseBinaryOp *op,
    ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const ::ttnn::Tensor &,
        const std::optional<const ::ttnn::DataType> &,
        const std::optional<::ttnn::MemoryConfig> &,
        std::optional<::ttnn::Tensor>,
        tt::stl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
        tt::stl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
        tt::stl::Span<const ::ttnn::operations::unary::UnaryWithParam>,
        std::optional<bool>)> &ttnnOp) {

  ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
  ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

  if (operations::utils::shouldSwapBinaryOperands(*lhs, *rhs)) {
    std::swap(lhs, rhs);
  }

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
                              std::nullopt, {}, {}, {}, /* use_legacy */ false);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseBinaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseBinaryOpType::Add: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::add);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Multiply: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::multiply);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Subtract: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::subtract);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Equal: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::eq);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::NotEqual: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::ne);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterEqual: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::ge);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterThan: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::gt);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessEqual: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::le);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessThan: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::lt);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Divide: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::divide);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalAnd: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::logical_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalOr: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::logical_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalXor: {
    runEltwiseNGBinaryOp(op, tensorPool, ::ttnn::logical_xor);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
