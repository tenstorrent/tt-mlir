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

// Quasar binds no unary op family at all, so there is no quasar::relu or
// quasar::sigmoid to substitute: the mainline unary path builds a
// DataMovementKernel, whose constructor TT_FATALs there.
//
// Express the unary as a tensor-scalar add with the op fused as an LHS
// activation. f(x) + 0 == f(x), adding 0.0f is exact in bf16, and both
// tensor-scalar add and LHS activation fusion route onto Quasar's binary_ng
// FPU kernel -- the path already proven by the Add case in binary.cpp.
//
// ANY UnaryOpType whose init/func pair resolves on Quasar can be carried this
// way, which is why this is a helper rather than one inline lambda:
// add_activation_defines() (quasar/binary_ng/device/binary_ng_utils.cpp) feeds
// the type through unary::utils::get_op_init_and_func and emits the resulting
// *_tile_init / *_tile pair into the compute kernel, PLUS the type's include
// macro via update_macro_defines(); the Quasar compute kernels all #include
// eltwise_unary/sfpu_split_includes.h, which turns that macro into the one
// header carrying the pair, and honour the body through
// PROCESS_<operand>_ACTIVATIONS (kernels/compute/eltwise_utils.hpp).
//
// Resolution, traced per type through get_macro_definition() and
// sfpu_split_includes.h, and then to the Quasar LLK:
//   RELU    SFPU_OP_RELU_FAMILY_INCLUDE   -> relu.h
//   SIGMOID (default macro)               -> compute_kernel_api.h, where
//           sigmoid_tile_init/sigmoid_tile each carry an ARCH_QUASAR arm
//   SQRT    SFPU_OP_SQRT_INCLUDE          -> sqrt.h:17,34       (no guard)
//   NEG     SFPU_OP_NEG_INCLUDE           -> negative.h:32      (no guard)
//   SIN/COS SFPU_OP_TRIG_FAMILY_INCLUDE   -> trigonometry.h:18,35,48,65
//                                            (no guard)
// and each bottoms out in a real Quasar kernel: ckernel_sfpu_sqrt.h
// calculate_sqrt/sqrt_init, ckernel_sfpu_negative.h _calculate_negative_,
// ckernel_sfpu_trigonometry.h calculate_sine/sine_init, calculate_cosine/
// cosine_init. (log_tile_init, by contrast, sits under #ifndef ARCH_QUASAR in
// compute_kernel_api.h, which is why Softmax cannot be carried this way; the
// same #ifndef removes power_tile, so neither can Pow.)
//
// The activation is built with NO parameters, so get_op_init_and_func() takes
// its *default* arm (get_op_init_and_func_default) rather than the
// parameterized one. That matters beyond style for SQRT: SQRT is in
// is_parametrized_type(), and its parameterized arm emits
// sqrt_tile<FAST_APPROX>, while Quasar's calculate_sqrt static_asserts
// FAST_APPROX == false. The no-parameter default arm emits the plain
// sqrt_tile(i) and is the only form that compiles -- and it is also exactly
// what the mainline fast_and_approximate_mode=false case asks for.
//
// NEG IS DTYPE-RESTRICTED. get_op_init_and_func_default() emits
// negative_tile_int32() when the input dtype is INT32, and negative_tile_int32
// is #ifndef ARCH_QUASAR (negative.h:37) -- absent on Quasar even though the
// LLK's _calculate_negative_int_ is present. Only the float arm is carried;
// see the Neg case below. (int32 would in any case be refused by
// matches_metal_v2_slice, which keeps every int32 binary on the descriptor
// path because the Quasar DFB int32 compute returns all-zero/garbage tiles.)
//
// NUMERICS. f(x) + 0.0f is exact in IEEE-754 for every finite f(x): x + (+0)
// returns x bit-for-bit, in bf16 and in fp32, with the single exception that
// -0.0 + 0.0 == +0.0. The rewrite therefore cannot change a result except by
// normalising a negative zero, which no pcc or top-1 comparison can see. What
// the rewrite does NOT change, and does not claim to, is the accuracy of the
// SFPU kernel itself: sqrt/sin/cos on Quasar are whatever those LLKs compute,
// exactly as a native unary would be.
//
// A rewrite rather than a redirect, and deliberately a stopgap: folding the
// activation into the producing conv/add belongs in the compiler's fusing
// pass.
static void runQuasarUnaryAsBinaryActivation(
    const ::tt::target::ttnn::EltwiseUnaryOp *op, ProgramTensorPool &tensorPool,
    ::ttnn::operations::unary::UnaryOpType activationType) {
  runEltwiseUnaryOp(
      op, tensorPool,
      [activationType](
          const ::ttnn::Tensor &in,
          const std::optional<::ttnn::MemoryConfig> &memoryConfig,
          const std::optional<::ttnn::Tensor> &optionalOutputTensor,
          const std::optional<::ttnn::CoreRangeSet> &) {
        using ::ttnn::operations::unary::EltwiseUnaryWithParam;
        // Built with NO parameters on purpose: get_op_init_and_func() then
        // takes its default arm, which for SIGMOID is
        // sigmoid_tile_init(); sigmoid_tile(i) -- template-defaulted to
        // VectorMode::RC and the accurate (non fast_and_approx) variant,
        // i.e. what the mainline Sigmoid case below asks for explicitly.
        const std::array<EltwiseUnaryWithParam, 1> activation{
            EltwiseUnaryWithParam(activationType)};
        return ::ttnn::operations::experimental::quasar::binary::add(
            in, 0.0f, /*output_dtype=*/std::nullopt, memoryConfig,
            optionalOutputTensor, /*post_activations=*/{},
            /*lhs_activations=*/activation);
      });
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
    // Same rewrite as Relu/Sigmoid; COS resolves to trigonometry.h:48,65,
    // which has no ARCH_QUASAR guard.
    if (utils::isQuasar()) {
      runQuasarUnaryAsBinaryActivation(
          op, tensorPool, ::ttnn::operations::unary::UnaryOpType::COS);
    } else {
      runEltwiseUnaryOp(op, tensorPool, ::ttnn::cos);
    }
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
    // Same rewrite as Relu/Sigmoid, but FLOAT DTYPES ONLY. For an INT32 input
    // get_op_init_and_func_default emits negative_tile_int32(), which is
    // #ifndef ARCH_QUASAR (negative.h:37) and so does not exist on Quasar;
    // carrying int32 here would turn a clean "unported kernel" abort into a
    // kernel compile failure. int32 keeps the mainline path.
    if (utils::isQuasar() &&
        tensorPool.getTTNNTensorAndValidate(op->in()).dtype() !=
            ::ttnn::DataType::INT32) {
      runQuasarUnaryAsBinaryActivation(
          op, tensorPool, ::ttnn::operations::unary::UnaryOpType::NEG);
    } else {
      runEltwiseUnaryOp(op, tensorPool, ::ttnn::neg);
    }
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Relu: {
    if (utils::isQuasar()) {
      runQuasarUnaryAsBinaryActivation(
          op, tensorPool, ::ttnn::operations::unary::UnaryOpType::RELU);
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
    // Same rewrite as Relu/Sigmoid. The no-parameter activation takes
    // get_op_init_and_func's default arm, i.e. sqrt_tile_init(); sqrt_tile(i),
    // which is the FAST_APPROX=false form the mainline call below also
    // requests -- and the only form Quasar's calculate_sqrt accepts.
    if (utils::isQuasar()) {
      runQuasarUnaryAsBinaryActivation(
          op, tensorPool, ::ttnn::operations::unary::UnaryOpType::SQRT);
    } else {
      runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::sqrt);
    }
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Rsqrt: {
    runEltwiseUnaryWithFastAndApproximateModeOp(op, tensorPool, ::ttnn::rsqrt);
    break;
  }
  case ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid: {
    // Same rewrite as Relu: Quasar has no unary sigmoid, but SIGMOID is
    // expressible as a fused LHS activation on the binary_ng FPU kernel.
    if (utils::isQuasar()) {
      runQuasarUnaryAsBinaryActivation(
          op, tensorPool, ::ttnn::operations::unary::UnaryOpType::SIGMOID);
    } else {
      runEltwiseUnarySigmoidOp(op, tensorPool, ::ttnn::sigmoid);
    }
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
    // Same rewrite as Relu/Sigmoid; SIN resolves to trigonometry.h:18,35,
    // which has no ARCH_QUASAR guard.
    if (utils::isQuasar()) {
      runQuasarUnaryAsBinaryActivation(
          op, tensorPool, ::ttnn::operations::unary::UnaryOpType::SIN);
    } else {
      runEltwiseUnaryOp(op, tensorPool, ::ttnn::sin);
    }
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
  case ::tt::target::ttnn::EltwiseUnaryOpType::Round: {
    runEltwiseUnaryOp(
        op, tensorPool,
        [](const ::ttnn::Tensor &input,
           const std::optional<::ttnn::MemoryConfig> &memoryConfig,
           const std::optional<::ttnn::Tensor> &optionalOutput,
           const std::optional<::ttnn::CoreRangeSet> &subCoreGrids) {
          return ::ttnn::round(input, /*parameter=*/std::nullopt, memoryConfig,
                               optionalOutput, subCoreGrids);
        });
    break;
  }
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::unary
