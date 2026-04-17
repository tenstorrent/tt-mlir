// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary_composite.h"
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

struct EltwiseBinaryCompositeOpSetup {
  ::ttnn::Tensor *lhs;
  ::ttnn::Tensor *rhs;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam> activations;
  std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam>
      inputTensorAActivations;
  std::vector<::ttnn::operations::unary::EltwiseUnaryWithParam>
      inputTensorBActivations;
};

static EltwiseBinaryCompositeOpSetup setupEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool) {
  ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
  ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  return {lhs,
          rhs,
          outputMemoryConfig,
          toTTNNUnaryWithParamVector(op->activations()),
          toTTNNUnaryWithParamVector(op->input_tensor_a_activations()),
          toTTNNUnaryWithParamVector(op->input_tensor_b_activations())};
}

template <typename Fn>
static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn &&ttnnOp) {
  auto setup = setupEltwiseBinaryCompositeOp(op, tensorPool);

  ::ttnn::Tensor out = ttnnOp(*setup.lhs, *setup.rhs, setup.outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

template <typename Fn>
static void runEltwiseBinaryCompositeOpWithActivations(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn &&ttnnOp) {
  auto setup = setupEltwiseBinaryCompositeOp(op, tensorPool);

  ::ttnn::Tensor out =
      ttnnOp(*setup.lhs, *setup.rhs, std::nullopt, setup.outputMemoryConfig,
             std::nullopt, setup.activations, setup.inputTensorAActivations,
             setup.inputTensorBActivations, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

template <typename Fn>
static void runEltwiseBinaryCompositeBitwiseOpWithActivations(
    const ::tt::target::ttnn::EltwiseBinaryCompositeOp *op,
    ProgramTensorPool &tensorPool, Fn &&ttnnOp) {
  auto setup = setupEltwiseBinaryCompositeOp(op, tensorPool);

  ::ttnn::Tensor out =
      ttnnOp(*setup.lhs, *setup.rhs, setup.outputMemoryConfig, std::nullopt,
             setup.activations, setup.inputTensorAActivations,
             setup.inputTensorBActivations, std::nullopt);

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
    runEltwiseBinaryCompositeOpWithActivations(
        op, tensorPool, [](auto &&...args) {
          return ::ttnn::maximum(std::forward<decltype(args)>(args)...);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Minimum: {
    runEltwiseBinaryCompositeOpWithActivations(
        op, tensorPool, [](auto &&...args) {
          return ::ttnn::minimum(std::forward<decltype(args)>(args)...);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::LogicalLeftShift: {
    runEltwiseBinaryCompositeBitwiseOpWithActivations(
        op, tensorPool, [](auto &&...args) {
          return ::ttnn::logical_left_shift(
              std::forward<decltype(args)>(args)...);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Remainder: {
    runEltwiseBinaryCompositeOpWithActivations(
        op, tensorPool, [](auto &&...args) {
          return ::ttnn::remainder(std::forward<decltype(args)>(args)...);
        });
    break;
  }
  case ::tt::target::ttnn::EltwiseBinaryCompositeOpType::Pow: {
    runEltwiseBinaryCompositeOpWithActivations(
        op, tensorPool, [](auto &&...args) {
          return ::ttnn::pow(std::forward<decltype(args)>(args)...);
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

void run(
    const ::tt::target::ttnn::EltwiseBinaryCompositeWithoutFusedActivationOp
        *op,
    ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseBinaryCompositeWithoutFusedActivationOpType::
      Atan2: {
    ::ttnn::Tensor *lhs = &(tensorPool.getTTNNTensorAndValidate(op->lhs()));
    ::ttnn::Tensor *rhs = &(tensorPool.getTTNNTensorAndValidate(op->rhs()));

    std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
        ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
            op->memory_config());
    LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                   outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");

    ::ttnn::Tensor out = ::ttnn::atan2(*lhs, *rhs, outputMemoryConfig);

    tensorPool.insertTTNNTensorAndValidate(op->out(), out);
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
