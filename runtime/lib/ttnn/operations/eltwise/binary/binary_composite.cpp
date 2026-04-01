// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary_composite.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

template <typename Fn>
static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn &&ttnnOp) {

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

static void
runPowScalarOp(const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOp *op,
               auto &&exponent, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor *input = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::pow(*input, exponent, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

// Handles the binary composite ops with LHS=tensor and RHS=tensor.
void run(const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Maximum: {
    runEltwiseBinaryCompositeOp(
        op, tensorPool,
        [](const ::ttnn::Tensor &lhs, const ::ttnn::Tensor &rhs,
           const std::optional<::ttnn::MemoryConfig> &memCfg) {
          return ::ttnn::maximum(lhs, rhs, std::nullopt, memCfg);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Minimum: {
    runEltwiseBinaryCompositeOp(
        op, tensorPool,
        [](const ::ttnn::Tensor &lhs, const ::ttnn::Tensor &rhs,
           const std::optional<::ttnn::MemoryConfig> &memCfg) {
          return ::ttnn::minimum(lhs, rhs, std::nullopt, memCfg);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::LogicalLeftShift: {
    runEltwiseBinaryCompositeOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::logical_left_shift(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Remainder: {
    runEltwiseBinaryCompositeOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::remainder(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Pow: {
    runEltwiseBinaryCompositeOp(
        op, tensorPool,
        [](const ::ttnn::Tensor &lhs, const ::ttnn::Tensor &rhs,
           const std::optional<::ttnn::MemoryConfig> &memCfg) {
          return ::ttnn::pow(lhs, rhs, std::nullopt, memCfg);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Atan2: {
    runEltwiseBinaryCompositeOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::atan2(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseAnd: {
    runEltwiseBinaryCompositeOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::bitwise_and(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseOr: {
    runEltwiseBinaryCompositeOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::bitwise_or(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseXor: {
    runEltwiseBinaryCompositeOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::bitwise_xor(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  }
}

// Handles the binary composite ops with LHS=tensor and RHS=scalar.
void run(const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOp *op,
         ProgramContext &context) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpType::PowScalar: {
    switch (op->rhs_type()) {
    case ::tt::target::ttnn::NumberType::FP:
      runPowScalarOp(op, op->rhs_as_FP()->value(), context);
      break;
    case ::tt::target::ttnn::NumberType::I32:
      runPowScalarOp(op, op->rhs_as_I32()->value(), context);
      break;
    default:
      LOG_FATAL("unknown exponent type");
    }
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
