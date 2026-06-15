// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/moe_compute.h"
#include "tt/runtime/detail/common/logger.h"

#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute.hpp"

namespace tt::runtime::ttnn::operations::ccl {

namespace {

::ttnn::experimental::prim::detail::MoEActivationFunction
toMetalActivation(::tt::target::MoEActivationFunction fb) {
  switch (fb) {
  case ::tt::target::MoEActivationFunction::Silu:
    return ::ttnn::experimental::prim::detail::MoEActivationFunction::SILU;
  case ::tt::target::MoEActivationFunction::SwiGLU:
    return ::ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU;
  }
  LOG_FATAL("Unknown MoEActivationFunction enum value: ", static_cast<int>(fb));
  return ::ttnn::experimental::prim::detail::MoEActivationFunction::SILU;
}

} // namespace

void run(const ::tt::target::ttnn::MoeComputeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &tilizeInput =
      tensorPool.getTTNNTensorAndValidate(op->tilize_input_tensor());
  const ::ttnn::Tensor &tilizeIndices =
      tensorPool.getTTNNTensorAndValidate(op->tilize_expert_indices_tensor());
  const ::ttnn::Tensor &tilizeScores =
      tensorPool.getTTNNTensorAndValidate(op->tilize_expert_scores_tensor());
  const ::ttnn::Tensor &tilizeMapping =
      tensorPool.getTTNNTensorAndValidate(op->tilize_expert_mapping_tensor());
  const ::ttnn::Tensor &w0w1 =
      tensorPool.getTTNNTensorAndValidate(op->matmul_w0_w1_tensor());
  const ::ttnn::Tensor &w2 =
      tensorPool.getTTNNTensorAndValidate(op->matmul_w2_tensor());

  // Only the compute_only path is supported (compiler verifier enforces it):
  // the A2A combine is bypassed, the combine-path inputs stay unset, tt-metal
  // returns only the first five tensors, and matmul_output is the final output.
  LOG_ASSERT(op->compute_only(),
             "moe_compute runtime only supports compute_only=true");

  std::optional<uint32_t> bhRingSize;
  if (op->bh_ring_size()) {
    bhRingSize = op->bh_ring_size().value();
  }

  // compute_only: every combine-path input stays unset; tt-metal returns the
  // first five tensors and matmul_output is the final output.
  std::vector<::ttnn::Tensor> results = ::ttnn::experimental::moe_compute(
      tilizeInput, tilizeIndices, tilizeScores, tilizeMapping, w0w1, w2,
      op->layer_id(), op->output_height_shard_dim(), op->intermediate_size(),
      op->has_bias(), /*cluster_axis=*/std::nullopt, /*topology=*/std::nullopt,
      /*num_links=*/std::nullopt, /*mux_core_range_set=*/std::nullopt,
      /*output_memory_config=*/std::nullopt,
      /*optional_output_tensor=*/std::nullopt,
      /*cross_device_semaphore=*/std::nullopt,
      toMetalActivation(op->activation_function()), /*compute_only=*/true,
      bhRingSize);

  LOG_ASSERT(results.size() == 5, "moe_compute returned ", results.size(),
             " tensors; expected 5 (compute_only)");

  tensorPool.insertTTNNTensorAndValidate(op->per_expert_total_tokens(),
                                         results[0]);
  tensorPool.insertTTNNTensorAndValidate(op->expert_activation(), results[1]);
  tensorPool.insertTTNNTensorAndValidate(op->expert_to_token(), results[2]);
  tensorPool.insertTTNNTensorAndValidate(op->tilize_output(), results[3]);
  tensorPool.insertTTNNTensorAndValidate(op->matmul_output(), results[4]);
  // In compute_only there is no combine output; the flatbuffer's
  // combine_output ref aliases matmul_output (results[4]).
  tensorPool.insertTTNNTensorAndValidate(op->combine_output(), results[4]);
}
} // namespace tt::runtime::ttnn::operations::ccl
