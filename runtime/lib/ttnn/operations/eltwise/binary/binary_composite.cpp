// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary_composite.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryCompositeOp.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

template <typename Fn>
static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn &&ttnnOp) {

  ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
  ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

  target::ttnn::EltwiseBinaryCompositeOpT eltwiseBinaryCompositeOpNative;
  op->UnPackTo(&eltwiseBinaryCompositeOpNative);

  ttnn_op_invoke::EltwiseBinaryCompositeOpResult result =
      ttnn_op_invoke::callEltwiseBinaryComposite(
          ttnn_op_invoke::CallType::EXECUTE, eltwiseBinaryCompositeOpNative,
          std::forward<Fn>(ttnnOp), lhs, rhs);

  LOG_ASSERT(
      std::holds_alternative<::ttnn::Tensor>(result),
      "Expected output Tensor from callEltwiseBinaryComposite execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

static void
runPowScalarOp(const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOp *op,
               ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor *input = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));

  target::ttnn::EltwiseBinaryCompositeScalarOpT
      eltwiseBinaryCompositeScalarOpNative;
  op->UnPackTo(&eltwiseBinaryCompositeScalarOpNative);

  ttnn_op_invoke::EltwiseBinaryCompositeScalarOpResult result =
      ttnn_op_invoke::callEltwiseBinaryCompositeScalar(
          ttnn_op_invoke::CallType::EXECUTE,
          eltwiseBinaryCompositeScalarOpNative, input);

  LOG_ASSERT(
      std::holds_alternative<::ttnn::Tensor>(result),
      "Expected output Tensor from callEltwiseBinaryCompositeScalar execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
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
    runEltwiseBinaryCompositeOp(op, tensorPool,
                                WRAP_OP(::ttnn::logical_left_shift));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Remainder: {
    runEltwiseBinaryCompositeOp(
        op, tensorPool,
        [](const ::ttnn::Tensor &lhs, const ::ttnn::Tensor &rhs,
           const std::optional<::ttnn::MemoryConfig> &memCfg) {
          return ::ttnn::remainder(lhs, rhs, std::nullopt, memCfg);
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
    runEltwiseBinaryCompositeOp(op, tensorPool, WRAP_OP(::ttnn::atan2));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseAnd: {
    runEltwiseBinaryCompositeOp(op, tensorPool, WRAP_OP(::ttnn::bitwise_and));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseOr: {
    runEltwiseBinaryCompositeOp(op, tensorPool, WRAP_OP(::ttnn::bitwise_or));
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::BitwiseXor: {
    runEltwiseBinaryCompositeOp(op, tensorPool, WRAP_OP(::ttnn::bitwise_xor));
    break;
  }
  }
}

// Handles the binary composite ops with LHS=tensor and RHS=scalar.
void run(const ::tt::target::ttnn::EltwiseBinaryCompositeScalarOp *op,
         ProgramContext &context) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpType::PowScalar: {
    runPowScalarOp(op, context);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
