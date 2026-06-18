// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/prepare_moe_compute_w2_weights.h"

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

  uint32_t bhRingSize = op->bh_ring_size() ? op->bh_ring_size().value() : 12;
  ::ttnn::Tensor packed =
      hasBias ? ::ttnn::experimental::prepare_w2_tensor_with_bias(
                    w2, *bias2, L, E, op->intermediate_size(),
                    op->hidden_size(), bhRingSize)
              : ::ttnn::experimental::prepare_w2_tensor_for_moe_compute(
                    w2, L, E, op->intermediate_size(), op->hidden_size(),
                    bhRingSize);

  ::ttnn::Tensor out = ::ttnn::experimental::quantize_weights_via_host(
      packed, ::tt::tt_metal::DataType::BFLOAT4_B,
      ::ttnn::experimental::get_weight_mem_configs(
          &meshDevice, L, E, op->hidden_size(), op->intermediate_size(),
          hasBias, bhRingSize)
          .w2);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
