// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/unary/unary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/experimental/quasar/binary/binary.hpp"

namespace tt::runtime::ttnn::operations::eltwise::unary {

static void runEltwiseUnaryOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::Tensor> &,
        const std::optional<::ttnn::CoreRangeSet> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ttnnOp(in, outputMemoryConfig, /*optional_output_tensor=*/std::nullopt,
             /*sub_core_grids=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryTanhOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, const std::optional<::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::Tensor> &, bool,
        const std::optional<::ttnn::CoreRangeSet> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out =
      ttnnOp(in, outputMemoryConfig, /*optional_output_tensor=*/std::nullopt,
             /*approx=*/false, /*sub_core_grids=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryWithFastAndApproximateModeOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const bool,
                       const std::optional<::ttnn::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &,
                       const std::optional<::ttnn::CoreRangeSet> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(in, /*parameter=*/false, outputMemoryConfig,
                              /*optional_output_tensor=*/std::nullopt,
                              /*sub_core_grids=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnarySigmoidOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const int,
                       const ::ttnn::operations::unary::SigmoidMode,
                       const std::optional<::ttnn::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &,
                       const std::optional<::ttnn::CoreRangeSet> &)> &ttnnOp) {

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  auto sigmoidMode = ::ttnn::operations::unary::SigmoidMode::ACCURATE;
  ::ttnn::Tensor out = ttnnOp(
      in, static_cast<int>(::ttnn::operations::unary::VecMode::RC), sigmoidMode,
      outputMemoryConfig, /*optional_output_tensor=*/std::nullopt,
      /*sub_core_grids=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runEltwiseUnaryWithFloatParameterOp(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, float,
                       const std::optional<::ttnn::MemoryConfig> &,
                       const std::optional<::ttnn::Tensor> &,
                       const std::optional<::ttnn::CoreRangeSet> &)> &ttnnOp) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  float parameter = op->params_as_EltwiseOpWithFloatParams()->parameter();
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(in, parameter, outputMemoryConfig,
                              /*optional_output_tensor=*/std::nullopt,
                              /*sub_core_grids=*/std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::EltwiseUnaryOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseUnaryOpType::Abs: {
    runEltwiseUnaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::abs(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Ceil: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::ceil);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Cos: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::cos);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Acos: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::acos);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Floor: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::floor);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Gelu: {
    runEltwiseUnaryWithFastAndApproximateModeOp(
        op, tensorPool, [](auto &&...args) {
          return ::ttnn::gelu(std::forward<decltype(args)>(args)...);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::IsFinite: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::isfinite);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LogicalNot: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::logical_not);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Neg: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::neg);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Relu: {
    // Quasar has no unary op family at all under experimental/quasar/ -- only
    // binary and binary_ng -- so there is nothing to dispatch relu to. Express
    // it as a binary instead: max(x, 0) is exactly relu, and the Quasar binary
    // front-end has a tensor-scalar maximum. This is the only op in the
    // ResNet-50 graph that needs rewriting rather than redirecting (16 sites).
    //
    // Worth knowing if this regresses: QUASAR_PARITY_GAPS.md lists maximum as
    // LLK-capable but not covered by an op test on the no-broadcast path, so it
    // is plausible-but-unproven territory. The fallback is
    // add(x, 0, lhs_activations={relu}) -- tensor-scalar add and lhs activation
    // fusion are both inside the validated slice. The hand-written Quasar
    // ResNet-50 avoids the question entirely by folding relu into the preceding
    // add/conv as an activation, which is also the better answer for
    // performance and belongs in the compiler's fusing pass.
    if (utils::isQuasar()) {
      runEltwiseUnaryOp(
          op, tensorPool,
          [](const ::ttnn::Tensor &in,
             const std::optional<::ttnn::MemoryConfig> &memoryConfig,
             const std::optional<::ttnn::Tensor> &optionalOutputTensor,
             const std::optional<::ttnn::CoreRangeSet> &) {
            // max(x, 0) would be the obvious spelling, but Quasar's
            // tensor-scalar maximum is implemented on the unary clamp path and
            // so delegates to mainline ttnn::prim::unary -- measured, and
            // exactly what QUASAR_PARITY_GAPS.md priority (3) asks Metal to
            // reroute onto invoke_binary_ng. Use add(x, 0) with relu fused as an
            // LHS activation instead: relu(x) + 0 == relu(x), tensor-scalar add
            // and LHS activation fusion are both inside the validated slice,
            // and adding 0.0f is exact in bf16.
            const std::array<::ttnn::operations::unary::EltwiseUnaryWithParam, 1>
                reluActivation{::ttnn::operations::unary::EltwiseUnaryWithParam(
                    ::ttnn::operations::unary::UnaryOpType::RELU)};
            return ::ttnn::operations::experimental::quasar::binary::add(
                in, 0.0f, /*dtype=*/std::nullopt, memoryConfig,
                optionalOutputTensor, /*post_activations=*/{},
                /*lhs_activations=*/reluActivation);
          });
    } else {
      runEltwiseUnaryOp(op, tensorPool, ::ttnn::relu);
    }
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Relu6: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::relu6);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Hardsigmoid: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::hardsigmoid);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::sqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Rsqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::rsqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid: {
    runEltwiseUnarySigmoidOp(op, tensorPool, ::ttnn::sigmoid);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Silu: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::silu);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Mish: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::mish);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sin: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sin);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Asin: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::asin);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Asinh: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::asinh);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Reciprocal: {
    runEltwiseUnaryOp(op, tensorPool, [](auto &&...args) {
      return ::ttnn::reciprocal(std::forward<decltype(args)>(args)...);
    });
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sign: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::sign);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tan: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::tan);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Tanh: {
    runEltwiseUnaryTanhOp(op, tensorPool, ::ttnn::tanh);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Atan: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::atan);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Exp: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::exp);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Log: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::log);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Expm1: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::expm1);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu: {
    runEltwiseUnaryWithFloatParameterOp(op, tensorPool, ::ttnn::leaky_relu);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::BitwiseNot: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::bitwise_not);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Erf: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::erf);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Erfc: {
    runEltwiseUnaryOp(op, tensorPool, ::ttnn::erfc);
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary
