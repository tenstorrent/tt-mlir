// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/unary/unary.h"
#include "eltwise/unary/unifiedEltwiseUnaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "utils/utils.h"

namespace tt::runtime::ttnn::operations::eltwise::unary {

template <typename Fn>
static void runEltwiseUnaryOp(const ::tt::target::ttnn::EltwiseUnaryOp *op,
                              ProgramTensorPool &tensorPool, Fn ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpT;
  op->UnPackTo(&eltwiseUnaryOpT);

  unifiedOpLib::EltwiseUnaryOpResult result = unifiedOpLib::callEltwiseUnary(
      unifiedOpLib::CallType::EXECUTE, eltwiseUnaryOpT,
      std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnary execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

template <typename Fn>
static void runEltwiseUnaryTanhOp(const ::tt::target::ttnn::EltwiseUnaryOp *op,
                                  ProgramTensorPool &tensorPool, Fn ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpT;
  op->UnPackTo(&eltwiseUnaryOpT);

  unifiedOpLib::EltwiseUnaryOpResult result =
      unifiedOpLib::callEltwiseUnaryTanh(unifiedOpLib::CallType::EXECUTE,
                                         eltwiseUnaryOpT,
                                         std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnaryTanh execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

template <typename Fn>
static void runEltwiseUnaryWithFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    Fn ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpT;
  op->UnPackTo(&eltwiseUnaryOpT);

  unifiedOpLib::EltwiseUnaryOpResult result =
      unifiedOpLib::callEltwiseUnaryWithFastAndApproximateMode(
          unifiedOpLib::CallType::EXECUTE, eltwiseUnaryOpT,
          std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from "
             "callEltwiseUnaryWithFastAndApproximateMode execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

template <typename Fn>
static void
runEltwiseUnarySigmoidOp(const ::tt::target::ttnn::EltwiseUnaryOp *op,
                         ProgramTensorPool &tensorPool, Fn ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpT;
  op->UnPackTo(&eltwiseUnaryOpT);

  unifiedOpLib::EltwiseUnaryOpResult result =
      unifiedOpLib::callEltwiseUnarySigmoid(unifiedOpLib::CallType::EXECUTE,
                                            eltwiseUnaryOpT,
                                            std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnarySigmoid execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

template <typename Fn>
static void runEltwiseUnaryWithFloatParameterOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    Fn ttnnOp) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpT;
  op->UnPackTo(&eltwiseUnaryOpT);

  unifiedOpLib::EltwiseUnaryOpResult result =
      unifiedOpLib::callEltwiseUnaryWithFloatParameter(
          unifiedOpLib::CallType::EXECUTE, eltwiseUnaryOpT,
          std::forward<Fn>(ttnnOp), &in);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseUnaryWithFloatParameter "
             "execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseUnaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseUnaryOpType::Abs: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::abs));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Ceil: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::ceil));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Cos: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::cos));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Acos: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::acos));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Floor: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::floor));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Gelu: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::gelu));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::IsFinite: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::isfinite));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LogicalNot: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::logical_not));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Neg: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::neg));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Relu: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::relu));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Relu6: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::relu6));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Hardsigmoid: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::hardsigmoid));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::sqrt));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Rsqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::rsqrt));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid: {
    runEltwiseUnarySigmoidOp(op, tensorPool, WRAP_OP(::ttnn::sigmoid));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Silu: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::silu));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Mish: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::mish));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sin: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::sin));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Asin: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::asin));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Asinh: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::asinh);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Reciprocal: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::reciprocal));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sign: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::sign));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tan: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::tan));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tanh: {
    runEltwiseUnaryTanhOp(op, tensorPool, WRAP_OP(::ttnn::tanh));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Atan: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::atan));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::exp));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Log: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::log));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Expm1: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::expm1));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu: {
    runEltwiseUnaryWithFloatParameterOp(op, tensorPool,
                                        WRAP_OP(::ttnn::leaky_relu));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::BitwiseNot: {
    runEltwiseUnaryOp(op, tensorPool, WRAP_OP(::ttnn::bitwise_not));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Erf: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::erf));
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Erfc: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool,
                                                WRAP_OP(::ttnn::erfc));
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary
