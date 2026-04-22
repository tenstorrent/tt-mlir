// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/unary/unary_composite.h"
#include "eltwise/unary/unifiedEltwiseUnaryCompositeOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

namespace tt::runtime::ttnn::operations::eltwise::unary {

template <typename Fn>
static void runEltwiseUnaryCompositeOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpT;
  op->UnPackTo(&eltwiseUnaryCompositeOpT);

  unifiedOpLib::EltwiseUnaryCompositeOpResult result =
      unifiedOpLib::callEltwiseUnaryComposite(unifiedOpLib::CallType::EXECUTE,
                                              eltwiseUnaryCompositeOpT,
                                              std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnaryComposite execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

static void runEltwiseUnaryCompositeClampScalarOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpT;
  op->UnPackTo(&eltwiseUnaryCompositeOpT);

  unifiedOpLib::EltwiseUnaryCompositeOpResult result =
      unifiedOpLib::callEltwiseUnaryCompositeClampScalar(
          unifiedOpLib::CallType::EXECUTE, eltwiseUnaryCompositeOpT, &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnaryCompositeClampScalar "
             "execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

static void runEltwiseUnaryCompositeClampTensorOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttnn::Tensor min = tensorPool.getTTNNTensorAndValidate(
      op->params_as_ClampTensorOpParams()->min());
  ::ttnn::Tensor max = tensorPool.getTTNNTensorAndValidate(
      op->params_as_ClampTensorOpParams()->max());

  target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpT;
  op->UnPackTo(&eltwiseUnaryCompositeOpT);

  unifiedOpLib::EltwiseUnaryCompositeOpResult result =
      unifiedOpLib::callEltwiseUnaryCompositeClampTensor(
          unifiedOpLib::CallType::EXECUTE, eltwiseUnaryCompositeOpT, &in, &min,
          &max);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnaryCompositeClampTensor "
             "execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

template <typename Fn>
static void runEltwiseUnaryCompositeWithFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpT;
  op->UnPackTo(&eltwiseUnaryCompositeOpT);

  unifiedOpLib::EltwiseUnaryCompositeOpResult result =
      unifiedOpLib::callEltwiseUnaryCompositeWithFastAndApproximateMode(
          unifiedOpLib::CallType::EXECUTE, eltwiseUnaryCompositeOpT,
          std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from "
             "callEltwiseUnaryCompositeWithFastAndApproximateMode execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseUnaryCompositeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Cbrt: {
    runEltwiseUnaryCompositeOp(op, tensorPool, WRAP_OP(::ttnn::cbrt));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar: {
    runEltwiseUnaryCompositeClampScalarOp(op, tensorPool);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor: {
    runEltwiseUnaryCompositeClampTensorOp(op, tensorPool);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryCompositeOpType::Log1p: {
    runEltwiseUnaryCompositeWithFastAndApproximateModeOp(
        op, tensorPool, WRAP_OP(::ttnn::log1p));
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary
