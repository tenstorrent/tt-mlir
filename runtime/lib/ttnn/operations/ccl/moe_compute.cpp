// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/moe_compute.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

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

  // Full A2A path: cluster_axis and mux_core_range_set are required;
  // num_links/topology auto-resolve from the fabric when null. The combine
  // output buffer and cross-device semaphore are bound below (leaving either
  // unbound takes tt-metal's use_init_semaphore path, which deadlocks).
  LOG_ASSERT(op->cluster_axis(), "moe_compute requires cluster_axis");
  uint32_t clusterAxis = op->cluster_axis().value();
  std::optional<::tt::tt_fabric::Topology> topology;
  if (op->topology()) {
    topology = ::tt::runtime::common::toMetalTopology(op->topology().value());
  }
  std::optional<uint32_t> numLinks = op->num_links();
  LOG_ASSERT(op->mux_core_range_set(),
             "moe_compute requires mux_core_range_set");
  ::ttnn::CoreRangeSet muxCoreRangeSet =
      ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*op->mux_core_range_set());

  // The combine semaphore and output buffer are allocated by IR ops and passed
  // in as operands; read them from the pools. Both must be bound or tt-metal
  // deadlocks the A2A combine.
  LOG_ASSERT(op->cross_device_semaphore(),
             "moe_compute requires cross_device_semaphore operand");
  ::ttnn::GlobalSemaphore &combineSemaphore =
      context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
          op->cross_device_semaphore());
  LOG_ASSERT(op->optional_output_tensor(),
             "moe_compute requires optional_output_tensor operand");
  const ::ttnn::Tensor &combineOutput =
      tensorPool.getTTNNTensorAndValidate(op->optional_output_tensor());

  std::vector<::ttnn::Tensor> results = ::ttnn::experimental::moe_compute(
      tilizeInput, tilizeIndices, tilizeScores, tilizeMapping, w0w1, w2,
      op->layer_id(), op->output_height_shard_dim(), op->intermediate_size(),
      op->has_bias(), clusterAxis, topology, numLinks, muxCoreRangeSet,
      /*output_memory_config=*/std::nullopt, combineOutput, combineSemaphore,
      toMetalActivation(op->activation_function()),
      /*compute_only=*/false);

  LOG_ASSERT(results.size() == 6, "moe_compute returned ", results.size(),
             " tensors; expected 6");

  // Only the combine output (results[5]) is exposed; free tt-metal's
  // routing/tilize/matmul intermediates (results[0..4]).
  tensorPool.insertTTNNTensorAndValidate(op->combine_output(), results[5]);
  for (size_t i = 0; i < 5; ++i) {
    ::ttnn::deallocate(results[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::ccl
