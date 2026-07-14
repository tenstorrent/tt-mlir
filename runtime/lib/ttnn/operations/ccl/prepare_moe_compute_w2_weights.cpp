// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/prepare_moe_compute_w2_weights.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PrepareMoEComputeW2WeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &w2 = tensorPool.getTTNNTensorAndValidate(op->w2());

  std::optional<::ttnn::Tensor> bias2;
  if (op->bias_2() != nullptr) {
    bias2 = tensorPool.getTTNNTensorAndValidate(op->bias_2());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  // tt-metal exposes no single weight-prep entry: pack W2, then bf4-quantize
  // into the bank-permuted DRAM HEIGHT_SHARDED config.
  uint32_t L = w2.logical_shape()[0];
  uint32_t E = w2.logical_shape()[1];
  bool hasBias = bias2.has_value();

  // Fast path (opt-in via TTXLA_MOE_FAST_QUANTIZE): on-device ttnn::typecast to
  // BFLOAT4_B instead of the default host round-trip quantize_weights_via_host.
  const bool fastQuantize = std::getenv("TTXLA_MOE_FAST_QUANTIZE") != nullptr;
  const auto memCfg = ::ttnn::experimental::get_weight_mem_configs(
                          &meshDevice, L, E, op->hidden_size(),
                          op->intermediate_size(), hasBias)
                          .w2;

  const auto tPrep0 = std::chrono::steady_clock::now();
  ::ttnn::Tensor packed =
      hasBias
          ? ::ttnn::experimental::prepare_w2_tensor_with_bias(
                w2, *bias2, L, E, op->intermediate_size(), op->hidden_size())
          : ::ttnn::experimental::prepare_w2_tensor_for_moe_compute(
                w2, L, E, op->intermediate_size(), op->hidden_size());
  const auto tPrep1 = std::chrono::steady_clock::now();

  // [EXPERIMENT] fast path in 2 steps: typecast to BFLOAT4_B keeping the input's
  // (interleaved) layout, then to_memory_config reshards to the HEIGHT_SHARDED
  // moe_compute weight layout (ttnn::typecast rejects an in/out layout change).
  ::ttnn::Tensor out =
      fastQuantize
          ? ::ttnn::to_memory_config(
                ::ttnn::typecast(packed, ::tt::tt_metal::DataType::BFLOAT4_B),
                memCfg)
          : ::ttnn::experimental::quantize_weights_via_host(
                packed, ::tt::tt_metal::DataType::BFLOAT4_B, memCfg);
  const auto tQuant1 = std::chrono::steady_clock::now();

  // Instrumentation (compile-time const-eval); fprintf bypasses loguru filter.
  std::fprintf(
      stderr,
      "[MOE_PREP_TIMING] w2   L=%u E=%u bias=%d fast=%d: prepare=%.1fs "
      "quantize=%.1fs total=%.1fs\n",
      L, E, static_cast<int>(hasBias), static_cast<int>(fastQuantize),
      std::chrono::duration<double>(tPrep1 - tPrep0).count(),
      std::chrono::duration<double>(tQuant1 - tPrep1).count(),
      std::chrono::duration<double>(tQuant1 - tPrep0).count());
  std::fflush(stderr);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
