// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/prepare_moe_compute_w0_w1_weights.h"

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

  uint32_t bhRingSize = op->bh_ring_size() ? op->bh_ring_size().value() : 12;
  ::ttnn::Tensor packed =
      hasBias ? ::ttnn::experimental::prepare_w0_w1_tensor_with_bias(
                    w0, w1, *bias0, *bias1, L, E, op->hidden_size(),
                    op->intermediate_size(), bhRingSize)
              : ::ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute(
                    w0, w1, L, E, op->hidden_size(), op->intermediate_size(),
                    bhRingSize);

  ::ttnn::Tensor out = ::ttnn::experimental::quantize_weights_via_host(
      packed, ::tt::tt_metal::DataType::BFLOAT4_B,
      ::ttnn::experimental::get_weight_mem_configs(
          &meshDevice, L, E, op->hidden_size(), op->intermediate_size(),
          hasBias, bhRingSize)
          .w0_w1);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
