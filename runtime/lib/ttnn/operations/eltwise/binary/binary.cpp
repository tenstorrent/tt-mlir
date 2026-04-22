// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary.h"
#include "eltwise/binary/unifiedEltwiseBinaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

template <typename Fn>
static void runEltwiseBinaryOp(const ::tt::target::ttnn::EltwiseBinaryOp *op,
                               ProgramTensorPool &tensorPool, Fn &&ttnnOp) {
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->lhs());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->rhs());

  target::ttnn::EltwiseBinaryOpT eltwiseBinaryOpT;
  op->UnPackTo(&eltwiseBinaryOpT);

  unifiedOpLib::EltwiseBinaryOpResult result = unifiedOpLib::callEltwiseBinary(
      unifiedOpLib::CallType::EXECUTE, eltwiseBinaryOpT,
      std::forward<Fn>(ttnnOp), &lhs, &rhs);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseBinary execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseBinaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseBinaryOpType::Add: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::add));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Multiply: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::multiply));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalRightShift: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::logical_right_shift));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Subtract: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::subtract));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Equal: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::eq));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::NotEqual: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::ne));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterEqual: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::ge));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::GreaterThan: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::gt));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessEqual: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::le));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LessThan: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::lt));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::Divide: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::divide));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalAnd: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::logical_and));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalOr: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::logical_or));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryOpType::LogicalXor: {
    runEltwiseBinaryOp(op, tensorPool, WRAP_OP(::ttnn::logical_xor));
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
