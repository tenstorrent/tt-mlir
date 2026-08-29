// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/experimental/quasar/binary/binary.hpp"

namespace tt::runtime::ttnn::operations::eltwise::binary {

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

// Run one eltwise binary op, selecting the Quasar entry point when the device
// is Quasar (see utils::isQuasar() for why the mainline ops cannot be used
// there). Every op dispatched through this macro has a Quasar counterpart in
// ttnn::operations::experimental::quasar::binary whose leading
// (lhs, rhs, output_dtype, memory_config) parameters match the mainline op's,
// so the forwarded argument pack binds to both.
#define RUN_ELTWISE_BINARY(NAME)                                               \
  runEltwiseBinaryOp(op, tensorPool, [](auto &&...args) {                      \
    return utils::isQuasar()                                                   \
               ? ::ttnn::operations::experimental::quasar::binary::NAME(       \
                     std::forward<decltype(args)>(args)...)                    \
               : ::ttnn::NAME(std::forward<decltype(args)>(args)...);          \
  })

void run(const ::tt::target::ttnn::EltwiseBinaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseBinaryOpType::Add: {
    RUN_ELTWISE_BINARY(add);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Multiply: {
    RUN_ELTWISE_BINARY(multiply);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalRightShift: {
    RUN_ELTWISE_BINARY(logical_right_shift);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Subtract: {
    RUN_ELTWISE_BINARY(subtract);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Equal: {
    RUN_ELTWISE_BINARY(eq);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::NotEqual: {
    RUN_ELTWISE_BINARY(ne);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterEqual: {
    RUN_ELTWISE_BINARY(ge);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterThan: {
    RUN_ELTWISE_BINARY(gt);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessEqual: {
    RUN_ELTWISE_BINARY(le);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessThan: {
    RUN_ELTWISE_BINARY(lt);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Divide: {
    RUN_ELTWISE_BINARY(divide);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalAnd: {
    RUN_ELTWISE_BINARY(logical_and);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalOr: {
    RUN_ELTWISE_BINARY(logical_or);
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalXor: {
    RUN_ELTWISE_BINARY(logical_xor);
    break;
  }
  }
}

#undef RUN_ELTWISE_BINARY

} // namespace tt::runtime::ttnn::operations::eltwise::binary
