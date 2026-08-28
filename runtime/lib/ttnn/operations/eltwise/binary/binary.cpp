// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

#include <string>
#include <vector>

namespace tt::runtime::ttnn::operations::eltwise::binary {

// MultiplyOp::verify already rejects anything but "silu" at compile time; this
// asserts because a dropped activation would be wrong numerics, silently.
static std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam>
lhsActivationsFromOp(const ::tt::target::ttnn::EltwiseBinaryOp *op) {
  std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam> acts;
  if (!op->lhs_activation() || op->lhs_activation()->size() == 0) {
    return acts;
  }
  const std::string activation = op->lhs_activation()->str();
  LOG_ASSERT(activation == "silu", "Unsupported lhs_activation \"", activation,
             "\" on eltwise binary op; only \"silu\" is supported");
  acts.emplace_back(::ttnn::operations::unary::UnaryOpType::SILU);
  return acts;
}

template <typename Fn>
static void runEltwiseBinaryOp(const ::tt::target::ttnn::EltwiseBinaryOp *op,
                               ProgramTensorPool &tensorPool, Fn &&ttnnOp) {

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

  ::ttnn::Tensor out = ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseBinaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  // The field is on the shared EltwiseBinaryOp table, so an op type other than
  // multiply can carry it and have it silently ignored.
  LOG_ASSERT(!op->lhs_activation() || op->lhs_activation()->size() == 0 ||
                 op->type() ==
                     ::tt::target::ttnn::EltwiseBinaryOpType::Multiply,
             "lhs_activation is only supported on multiply, but was set on ",
             ::tt::target::ttnn::EnumNameEltwiseBinaryOpType(op->type()));

  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseBinaryOpType::Add: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::add(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Multiply: {
    // ttnn takes these as non-owning spans, so both vectors must outlive the
    // call.
    std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam> noPostActs;
    std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam> lhsActs =
        lhsActivationsFromOp(op);
    runEltwiseBinaryOp(op, tensorPool, [&](auto &&...args) {
      return ::ttnn::multiply(std::forward<decltype(args)>(args)...,
                              /*optional_output=*/std::nullopt,
                              /*post_activations=*/noPostActs,
                              /*lhs_activations=*/lhsActs);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalRightShift: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::logical_right_shift(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Subtract: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::subtract(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Equal: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::eq(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::NotEqual: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::ne(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterEqual: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::ge(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterThan: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::gt(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessEqual: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::le(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessThan: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::lt(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Divide: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::divide(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalAnd: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::logical_and(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalOr: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::logical_or(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalXor: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::logical_xor(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
