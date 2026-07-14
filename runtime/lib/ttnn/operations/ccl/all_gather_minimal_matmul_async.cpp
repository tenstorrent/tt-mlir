// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_gather_minimal_matmul_async.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async.hpp"

namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllGatherMinimalMatmulAsyncOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &weight =
      tensorPool.getTTNNTensorAndValidate(op->weight());

  std::optional<::ttnn::Tensor> bias = std::nullopt;
  if (op->bias()) {
    bias = tensorPool.getTTNNTensorAndValidate(op->bias());
  }

  std::optional<float> scalar = std::nullopt;
  if (op->scalar()) {
    scalar = op->scalar().value();
  }

  std::optional<::ttnn::Tensor> addcmulInput1 = std::nullopt;
  if (op->addcmul_input1()) {
    addcmulInput1 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input1());
  }

  std::optional<::ttnn::Tensor> addcmulInput2 = std::nullopt;
  if (op->addcmul_input2()) {
    addcmulInput2 = tensorPool.getTTNNTensorAndValidate(op->addcmul_input2());
  }

  // The all-gather is synchronized by two multi-device global semaphores.
  std::vector<::ttnn::GlobalSemaphore> multiDeviceSemaphore;
  for (const auto *semaphoreRef : *op->multi_device_semaphore()) {
    multiDeviceSemaphore.push_back(
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            semaphoreRef));
  }

  std::optional<::ttnn::GlobalSemaphore> barrierSemaphore = std::nullopt;
  if (op->barrier_semaphore()) {
    barrierSemaphore =
        context.getGlobalSemaphorePool().getTTNNGlobalSemaphoreAndValidate(
            op->barrier_semaphore());
  }

  // Topology is a required metal parameter; default to Linear when unset.
  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear;
  if (op->topology()) {
    topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op->topology().value()));
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  std::optional<::ttnn::DataType> dtype = std::nullopt;
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*op->dtype());
  }

  uint32_t numLinks = op->num_links() ? op->num_links().value() : 1;

  std::optional<uint32_t> clusterAxis = std::nullopt;
  if (op->cluster_axis()) {
    clusterAxis = op->cluster_axis().value();
  }

  // `fused_activation`, the `MinimalMatmulConfig`, `compute_kernel_config`, the
  // persistent buffers and the FSDP path are not modeled by the compiler yet;
  // pass their tt-metal defaults.
  std::vector<::ttnn::Tensor> outputs = ::ttnn::all_gather_minimal_matmul_async(
      input, weight, bias, scalar, addcmulInput1, addcmulInput2,
      /*fused_activation=*/std::nullopt, /*config=*/std::nullopt,
      multiDeviceSemaphore, topology, memoryConfig, dtype,
      /*compute_kernel_config=*/std::nullopt,
      /*persistent_output_buffer=*/std::nullopt, numLinks, clusterAxis,
      barrierSemaphore, op->force_transpose(), op->num_workers_per_link(),
      op->num_buffers_per_channel(), op->chunks(), op->dim());

  const auto *outputRefs = op->outputs();
  LOG_ASSERT(outputs.size() == outputRefs->size(),
             "all_gather_minimal_matmul_async produced ", outputs.size(),
             " outputs but the flatbuffer expects ", outputRefs->size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    tensorPool.insertTTNNTensorAndValidate(outputRefs->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::ccl
