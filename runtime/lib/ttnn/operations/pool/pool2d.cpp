// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/pool2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"
#include <optional>
#include <ttnn/operations/functions.hpp>
#include <ttnn/operations/pool/generic/generic_pools.hpp>
#include "ttnn/operations/experimental/quasar/pool_generic/generic_pools.hpp"
#include "ttnn/operations/experimental/quasar/binary/binary_composite.hpp"
#include "ttnn/operations/experimental/quasar/pad/pad.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/slice/slice.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

#include <cstdlib>
#include <limits>
#include <vector>

namespace tt::runtime::ttnn::operations::pool {

namespace {
// TTIRToTTNN hardcodes config_tensors_in_dram=true for every pooling op
// (Pooling2dOpConversionPattern, TTIRToTTNN.cpp:2568/2576/2592) to avoid L1_SMALL
// pressure on Wormhole. On Quasar that routes into the DRAM-config halo path,
// whose gather kernel does not compile: halo_gather.cpp reads a per-core runtime
// arg into a constexpr. Quasar has 4 MB of L1 per core rather than Wormhole's
// 1.5 MB, so the reason for the override does not apply -- keep the config
// tensors in L1 and stay on the path that works.
template <typename OpT>
bool configTensorsInDram(const OpT *op) {
  return utils::isQuasar() ? false : op->config_tensors_in_dram();
}

// Max pooling as one slice per kernel tap, then a pairwise maximum.
//
// quasar::max_pool2d does not complete on the ZeBu emulator: traced with
// TTMLIR_OP_TRACE=1, the run reaches op 8 (the Pool2dOp) and sits there with no
// further program launch for over twenty minutes, against a measured ~30 s per
// launch. It is correct on craq-sim, so this is the emulator's halo/gather path
// rather than the maths.
//
// The decomposition is the one already used for convolution: pad in ROW_MAJOR,
// take a strided window per tap, and combine the taps with a primitive that is
// measured to work on both targets. For max pooling the combiner is
// quasar::maximum and the padding value has to be the identity of max, not zero
// -- zero padding would win over every negative activation at the border.
::ttnn::Tensor
maxPool2dViaTaps(const ::ttnn::Tensor &input, uint32_t N, uint32_t H, uint32_t W,
                 uint32_t C, const std::array<uint32_t, 2> &kernelSize,
                 const std::array<uint32_t, 2> &stride,
                 const std::array<uint32_t, 4> &pads, bool ceilMode,
                 const std::optional<::ttnn::MemoryConfig> &outputMemoryConfig,
                 ::ttnn::Layout outputLayout) {
  LOG_ASSERT(!ceilMode, "Quasar max-pool tap decomposition assumes floor mode");
  const uint32_t kh = kernelSize[0];
  const uint32_t kw = kernelSize[1];
  const uint32_t Hp = H + pads[0] + pads[1];
  const uint32_t Wp = W + pads[2] + pads[3];
  const uint32_t outH = (Hp - kh) / stride[0] + 1;
  const uint32_t outW = (Wp - kw) / stride[1] + 1;

  ::ttnn::Tensor act = input;
  if (act.layout() != ::ttnn::Layout::ROW_MAJOR) {
    act = ::ttnn::operations::experimental::quasar::to_layout(
        act, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
  }
  const std::vector<int32_t> spatialShape = {
      static_cast<int32_t>(N), static_cast<int32_t>(H), static_cast<int32_t>(W),
      static_cast<int32_t>(C)};
  act = ::ttnn::operations::experimental::quasar::reshape(act, spatialShape,
                                                          std::nullopt);
  if (pads[0] || pads[1] || pads[2] || pads[3]) {
    const ::ttsl::SmallVector<::ttnn::operations::experimental::quasar::PadSpecDim>
        padSpec = {{0u, 0u}, {pads[0], pads[1]}, {pads[2], pads[3]}, {0u, 0u}};
    // Not zero: the identity of max. bfloat16 carries this exactly, and every
    // real activation compares above it.
    act = ::ttnn::operations::experimental::quasar::pad(
        act, padSpec, -3.0e38f, /*use_multicore=*/true, std::nullopt,
        std::nullopt);
  }

  const std::vector<int32_t> flatShape = {
      1, 1, static_cast<int32_t>(N * outH * outW), static_cast<int32_t>(C)};
  std::vector<::ttnn::Tensor> taps;
  taps.reserve(static_cast<size_t>(kh) * kw);
  for (uint32_t i = 0; i < kh; i++) {
    for (uint32_t j = 0; j < kw; j++) {
      const std::vector<int32_t> begins = {
          0, static_cast<int32_t>(i), static_cast<int32_t>(j), 0};
      const std::vector<int32_t> ends = {
          static_cast<int32_t>(N),
          static_cast<int32_t>(i + (outH - 1) * stride[0] + 1),
          static_cast<int32_t>(j + (outW - 1) * stride[1] + 1),
          static_cast<int32_t>(C)};
      const std::vector<int32_t> steps = {
          1, static_cast<int32_t>(stride[0]), static_cast<int32_t>(stride[1]),
          1};
      ::ttnn::Tensor tap =
          ::ttnn::operations::experimental::quasar::slice<int32_t>(
              act, ::ttsl::Span<const int32_t>(begins),
              ::ttsl::Span<const int32_t>(ends),
              ::ttsl::Span<const int32_t>(steps), std::nullopt, std::nullopt,
              std::nullopt, std::nullopt);
      tap = ::ttnn::operations::experimental::quasar::reshape(tap, flatShape,
                                                              std::nullopt);
      // Tilize through logical data, never behind a launch -- see
      // utils::quasarTilizeNeedsHostRoute.
      if (utils::quasarTilizeNeedsHostRoute(tap, ::ttnn::Layout::TILE,
                                            std::nullopt)) {
        const std::vector<bfloat16> d = tap.to_vector<bfloat16>();
        const ::tt::tt_metal::TensorSpec spec(
            tap.logical_shape(),
            ::tt::tt_metal::TensorLayout(tap.dtype(), ::ttnn::Layout::TILE,
                                         tap.memory_config()));
        tap = ::ttnn::Tensor::from_vector(std::move(d), spec, tap.device());
      } else if (tap.layout() != ::ttnn::Layout::TILE) {
        tap = ::ttnn::operations::experimental::quasar::to_layout(
            tap, ::ttnn::Layout::TILE, std::nullopt, std::nullopt);
      }
      taps.push_back(tap);
    }
  }
  LOG_ASSERT(!taps.empty(), "Pooling window has no taps");
  // Pairwise, so the dependency chain is log2(n) deep rather than n. max is
  // exact in bf16, so this is about launch depth, not rounding.
  while (taps.size() > 1) {
    std::vector<::ttnn::Tensor> next;
    next.reserve((taps.size() + 1) / 2);
    for (size_t t = 0; t + 1 < taps.size(); t += 2) {
      next.push_back(::ttnn::operations::experimental::quasar::binary::maximum(
          taps[t], taps[t + 1]));
    }
    if (taps.size() % 2 == 1) {
      next.push_back(taps.back());
    }
    taps = std::move(next);
  }

  ::ttnn::Tensor out = taps.front();
  if (out.layout() != outputLayout) {
    out = ::ttnn::operations::experimental::quasar::to_layout(
        out, outputLayout, std::nullopt, outputMemoryConfig);
  }
  return out;
}

} // namespace


void runAvgPool2dOp(
    const ::tt::target::ttnn::Pool2dOp *op, ProgramTensorPool &tensorPool,
    const std::function<::ttnn::Tensor(
        const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>, bool,
        bool, std::optional<int32_t>,
        const std::optional<const ::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::operations::pool::Op2DSliceConfig> &,
        const std::optional<const ::ttnn::TensorMemoryLayout>,
        const std::optional<::ttnn::DeviceComputeKernelConfig> &, bool, bool,
        ::ttnn::DataType, ::ttnn::Layout, bool)> &ttnnOp) {
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op->padding()->Get(0)),
                                static_cast<uint32_t>(op->padding()->Get(1))};
  } else {
    padding = std::array<uint32_t, 4>{
        static_cast<uint32_t>(op->padding()->Get(0)),  // top
        static_cast<uint32_t>(op->padding()->Get(2)),  // bottom
        static_cast<uint32_t>(op->padding()->Get(1)),  // left
        static_cast<uint32_t>(op->padding()->Get(3))}; // right
  }

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  ::ttnn::Tensor out =
      ttnnOp(input, op->batch_size(), op->input_height(), op->input_width(),
             op->channels(), kernelSize, stride, padding, op->ceil_mode(),
             op->extra_params_as_AvgPool2dExtraParams()->count_include_pad(),
             /*divisor_override=*/std::nullopt, outputMemoryConfig,
             /*dram_slice_config=*/std::nullopt, appliedShardScheme,
             computeKernelConfig,
             /*deallocate_input=*/false,
             /*reallocate_halo_output=*/op->reallocate_halo_output(),
             ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
             /*config_tensor_in_dram=*/configTensorsInDram(op));

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void runMaxPool2dOp(
    const ::tt::target::ttnn::Pool2dOp *op, ProgramTensorPool &tensorPool,
    const std::function<std::vector<::ttnn::Tensor>(
        const ::ttnn::Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
        std::array<uint32_t, 2>, std::array<uint32_t, 2>,
        std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>,
        std::array<uint32_t, 2>, bool,
        const std::optional<const ::ttnn::MemoryConfig> &,
        const std::optional<::ttnn::operations::pool::Op2DSliceConfig> &,
        std::optional<const ::ttnn::TensorMemoryLayout>, bool, bool, bool,
        ::ttnn::DataType, ::ttnn::Layout, bool)> &ttnnOp) {
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op->padding()->Get(0)),
                                static_cast<uint32_t>(op->padding()->Get(1))};
  } else {
    padding = std::array<uint32_t, 4>{
        static_cast<uint32_t>(op->padding()->Get(0)),  // top
        static_cast<uint32_t>(op->padding()->Get(2)),  // bottom
        static_cast<uint32_t>(op->padding()->Get(1)),  // left
        static_cast<uint32_t>(op->padding()->Get(3))}; // right
  }

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  // Quasar: decompose rather than call the pool, which does not complete on the
  // emulator. Only the plain case -- no dilation, floor mode -- is covered;
  // anything else stays on the op so it fails loudly instead of quietly wrong.
  if (utils::isQuasar() && dilation[0] == 1 && dilation[1] == 1 &&
      !op->ceil_mode() && !std::getenv("TTMLIR_NO_POOL_TAPS")) {
    const std::array<uint32_t, 4> pads = std::visit(
        [](const auto &p) -> std::array<uint32_t, 4> {
          if constexpr (std::tuple_size_v<std::decay_t<decltype(p)>> == 2) {
            return {p[0], p[0], p[1], p[1]};
          } else {
            return {p[0], p[1], p[2], p[3]};
          }
        },
        padding);
    const ::ttnn::Layout outputLayout =
        ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
    ::ttnn::Tensor out = maxPool2dViaTaps(
        input, op->batch_size(), op->input_height(), op->input_width(),
        op->channels(), kernelSize, stride, pads, op->ceil_mode(),
        outputMemoryConfig, outputLayout);
    tensorPool.insertTTNNTensorAndValidate(op->out(), out);
    return;
  }

  std::vector<::ttnn::Tensor> results =
      ttnnOp(input, op->batch_size(), op->input_height(), op->input_width(),
             op->channels(), kernelSize, stride, padding, dilation,
             op->ceil_mode(), outputMemoryConfig,
             /*dram_slice_config=*/std::nullopt, appliedShardScheme,
             /*deallocate_input=*/false,
             /*reallocate_halo_output=*/op->reallocate_halo_output(),
             /*return_indices=*/false, ::ttnn::DataType::BFLOAT16,
             ::ttnn::Layout::ROW_MAJOR,
             /*config_tensor_in_dram=*/configTensorsInDram(op));

  tensorPool.insertTTNNTensorAndValidate(op->out(), results[0]);
}

void run(const ::tt::target::ttnn::Pool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::Pool2dOpType::AvgPool2d: {
    // Quasar's pools live in ttnn::operations::pool::quasar -- deliberately not
    // re-exported into ttnn:: to avoid colliding with the mainline names.
    // Same argument list, so this is a functor substitution.
    if (utils::isQuasar()) {
      runAvgPool2dOp(op, tensorPool,
                     ::ttnn::operations::pool::quasar::avg_pool2d);
    } else {
      runAvgPool2dOp(op, tensorPool, ::ttnn::avg_pool2d);
    }
    break;
  }
  case ::tt::target::ttnn::Pool2dOpType::MaxPool2d: {
    if (utils::isQuasar()) {
      runMaxPool2dOp(op, tensorPool,
                     ::ttnn::operations::pool::quasar::max_pool2d);
    } else {
      runMaxPool2dOp(op, tensorPool, ::ttnn::max_pool2d);
    }
  }
  }
}

void run(const ::tt::target::ttnn::MaxPool2dWithIndicesOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->result()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  std::array<uint32_t, 2> kernelSize, stride, dilation;
  std::copy_n(op->kernel_size()->begin(), 2, kernelSize.begin());
  std::copy_n(op->stride()->begin(), 2, stride.begin());
  std::copy_n(op->dilation()->begin(), 2, dilation.begin());

  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  if (op->padding()->size() == 2) {
    padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op->padding()->Get(0)),
                                static_cast<uint32_t>(op->padding()->Get(1))};
  } else {
    padding = std::array<uint32_t, 4>{
        static_cast<uint32_t>(op->padding()->Get(0)),  // top
        static_cast<uint32_t>(op->padding()->Get(2)),  // bottom
        static_cast<uint32_t>(op->padding()->Get(1)),  // left
        static_cast<uint32_t>(op->padding()->Get(3))}; // right
  }

  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme = std::nullopt;
  if (op->applied_shard_scheme()) {
    appliedShardScheme = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        *op->applied_shard_scheme());
  }

  // Call ttnn::max_pool2d with return_indices = true, returning both output and
  // indices. Use default BFLOAT16 dtype and ROW_MAJOR layout (required for
  // indices).
  // Bind the arguments once and let the caller pick the entry point; the Quasar
  // pool takes exactly the same parameters (generic_pools.hpp). A ternary over
  // function pointers does not compile -- the signatures are identical but the
  // declarations are distinct, so there is no common type.
  auto callMaxPool2d = [&](auto &&maxPoolFn) { return maxPoolFn(
      input, op->batch_size(), op->input_height(), op->input_width(),
      op->channels(), kernelSize, stride, padding, dilation, op->ceil_mode(),
      outputMemoryConfig, /*dram_slice_config=*/std::nullopt,
      appliedShardScheme,
      /*deallocate_input=*/false,
      /*reallocate_halo_output=*/op->reallocate_halo_output(),
      /*return_indices=*/true, ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR,
      /*config_tensor_in_dram=*/configTensorsInDram(op));
  };
  std::vector<::ttnn::Tensor> outputs =
      utils::isQuasar()
          ? callMaxPool2d(::ttnn::operations::pool::quasar::max_pool2d)
          : callMaxPool2d(::ttnn::max_pool2d);

  tensorPool.insertTTNNTensorAndValidate(op->result(), outputs[0]);
  tensorPool.insertTTNNTensorAndValidate(op->result_indices(), outputs[1]);
}

void run(const ::tt::target::ttnn::GlobalAvgPool2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::DataType dtype = input.dtype();
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*op->dtype());
  }

  auto inputShape = input.logical_shape();
  LOG_ASSERT(inputShape.rank() == 4,
             "GlobalAvgPool2d expects a rank 4 input tensor");
  uint32_t batchSize = inputShape[0];
  uint32_t inputHeight = inputShape[1];
  uint32_t inputWidth = inputShape[2];
  uint32_t inputChannels = inputShape[3];

  ::ttnn::Layout outputLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());

  auto callAvgPool2d = [&](auto &&avgPoolFn) { return avgPoolFn(
      input, batchSize, inputHeight, inputWidth, inputChannels,
      /*kernel_size=*/{inputHeight, inputWidth},
      /*stride=*/{1, 1}, /*padding=*/std::array<uint32_t, 2>{0, 0},
      /*ceil_mode=*/false, /*count_include_pad=*/true,
      /*divisor_override=*/std::nullopt, outputMemoryConfig,
      /*dram_slice_config=*/std::nullopt,
      /*applied_shard_scheme=*/std::nullopt,
      /*compute_kernel_config=*/std::nullopt,
      /*deallocate_input=*/false,
      /*reallocate_halo_output=*/true, dtype, outputLayout,
      /*config_tensor_in_dram=*/false);
  };
  ::ttnn::Tensor out =
      utils::isQuasar()
          ? callAvgPool2d(::ttnn::operations::pool::quasar::avg_pool2d)
          : callAvgPool2d(::ttnn::avg_pool2d);

  out = ::ttnn::reshape(out, ::ttnn::Shape({batchSize, 1, 1, inputChannels}),
                        outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::pool
