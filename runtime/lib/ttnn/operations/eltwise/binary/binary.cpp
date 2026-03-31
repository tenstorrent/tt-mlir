// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <vector>

namespace tt::runtime::ttnn::operations::eltwise::binary {

static std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam>
toTTNNUnaryWithParamVector(
    const flatbuffers::Vector<
        flatbuffers::Offset<::tt::target::ttnn::UnaryWithParam>> *activations) {
  std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam> converted;
  if (activations == nullptr) {
    return converted;
  }

  converted.reserve(activations->size());
  for (const auto *activation : *activations) {
    converted.push_back(
        ::tt::runtime::ttnn::operations::utils::toTTNNUnaryWithParam(
            *activation));
  }
  return converted;
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

  auto activations = toTTNNUnaryWithParamVector(op->activations());
  auto inputTensorAActivations =
      toTTNNUnaryWithParamVector(op->input_tensor_a_activations());
  auto inputTensorBActivations =
      toTTNNUnaryWithParamVector(op->input_tensor_b_activations());

  ::ttnn::Tensor out =
      ttnnOp(*lhs, *rhs, outputDataType, outputMemoryConfig, std::nullopt,
             activations, inputTensorAActivations, inputTensorBActivations);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseBinaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseBinaryOpType::Add: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::add(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Multiply: {
    runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::multiply(std::forward<decltype(args)>(args)...);
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
