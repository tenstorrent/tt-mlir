// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/prepare_moe_compute_w0_w1_weights.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::PrepareMoEComputeW0W1WeightsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &w0 = tensorPool.getTTNNTensorAndValidate(op->w0());
  const ::ttnn::Tensor &w1 = tensorPool.getTTNNTensorAndValidate(op->w1());

  std::optional<::ttnn::Tensor> bias0;
  if (op->bias_0() != nullptr) {
    bias0 = tensorPool.getTTNNTensorAndValidate(op->bias_0());
  }
  std::optional<::ttnn::Tensor> bias1;
  if (op->bias_1() != nullptr) {
    bias1 = tensorPool.getTTNNTensorAndValidate(op->bias_1());
  }

  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  // tt-metal exposes no single weight-prep entry: pack W0/W1, then bf4-quantize
  // into the bank-permuted DRAM HEIGHT_SHARDED config.
  uint32_t L = w0.logical_shape()[0];
  uint32_t E = w0.logical_shape()[1];
  bool hasBias = bias0.has_value();

  // Fast path (opt-in via TTXLA_MOE_FAST_QUANTIZE): on-device ttnn::typecast to
  // BFLOAT4_B instead of the default quantize_weights_via_host, which does a
  // from_device -> single-threaded host to_dtype -> to_device round-trip. The
  // host path is documented "slower but higher quality"; typecast is
  // faster/lower-quality.
  const bool fastQuantize = std::getenv("TTXLA_MOE_FAST_QUANTIZE") != nullptr;
  const auto memCfg = ::ttnn::experimental::get_weight_mem_configs(
                          &meshDevice, L, E, op->hidden_size(),
                          op->intermediate_size(), hasBias)
                          .w0_w1;

  const auto tPrep0 = std::chrono::steady_clock::now();
  ::ttnn::Tensor packed =
      hasBias ? ::ttnn::experimental::prepare_w0_w1_tensor_with_bias(
                    w0, w1, *bias0, *bias1, L, E, op->hidden_size(),
                    op->intermediate_size())
              : ::ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute(
                    w0, w1, L, E, op->hidden_size(), op->intermediate_size());
  const auto tPrep1 = std::chrono::steady_clock::now();

  ::ttnn::Tensor out =
      fastQuantize
          ? ::ttnn::typecast(packed, ::tt::tt_metal::DataType::BFLOAT4_B, memCfg)
          : ::ttnn::experimental::quantize_weights_via_host(
                packed, ::tt::tt_metal::DataType::BFLOAT4_B, memCfg);
  const auto tQuant1 = std::chrono::steady_clock::now();

  // Instrumentation: isolate the const-eval weight-prep cost (this runs during
  // compile-time const-eval). fprintf(stderr) bypasses loguru's level filter so
  // it is always visible under `pytest -s`.
  std::fprintf(
      stderr,
      "[MOE_PREP_TIMING] w0w1 L=%u E=%u bias=%d fast=%d: prepare=%.1fs "
      "quantize=%.1fs total=%.1fs\n",
      L, E, static_cast<int>(hasBias), static_cast<int>(fastQuantize),
      std::chrono::duration<double>(tPrep1 - tPrep0).count(),
      std::chrono::duration<double>(tQuant1 - tPrep1).count(),
      std::chrono::duration<double>(tQuant1 - tPrep0).count());
  std::fflush(stderr);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
