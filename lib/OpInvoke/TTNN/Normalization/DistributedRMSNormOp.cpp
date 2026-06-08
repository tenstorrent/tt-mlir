// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/DistributedRMSNormOp.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include "llvm/Support/ErrorHandling.h"

#include <functional>
#include <optional>

namespace ttnn_op_invoke {

DistributedRMSNormResolvedParams resolveDistributedRMSNormParams(
    const ::tt::target::ttnn::DistributedRMSNormOpT &op) {
  DistributedRMSNormResolvedParams params;
  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }
  if (op.program_config) {
    params.programConfig =
        operations::utils::createLayerNormShardedMultiCoreProgramConfig(
            *op.program_config);
  }
  if (op.sub_device_id.has_value()) {
    params.subDeviceId = std::make_optional<::tt::tt_metal::SubDeviceId>(
        op.sub_device_id.value());
  }
  if (op.num_links.has_value()) {
    params.numLinks = static_cast<size_t>(op.num_links.value());
  }
  if (op.topology.has_value()) {
    params.topology = static_cast<::ttnn::ccl::Topology>(
        ::tt::runtime::common::toMetalTopology(op.topology.value()));
  }
  params.dtype = std::nullopt;
  params.useNoc1Only = false;
  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createDistributedRMSNormTuple(
    Tag tag, const ::tt::target::ttnn::DistributedRMSNormOpT &op,
    TensorArg input, std::optional<TensorArg> residual_input_tensor,
    std::optional<TensorArg> weight, std::optional<TensorArg> stats,
    ::ttnn::MeshDevice *device, const ::ttnn::GlobalSemaphore &semaphore,
    const DistributedRMSNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag), params.programConfig, op.cluster_axis,
      std::cref(*device), semaphore,
      /*persistent_output_tensor=*/std::nullopt, params.numLinks,
      params.topology, params.subDeviceId, params.dtype, params.computeConfig,
      params.outputMemoryConfig,
      residual_input_tensor
          ? std::make_optional(resolveTensorArg(*residual_input_tensor, tag))
          : std::nullopt,
      op.epsilon,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      stats ? std::make_optional(resolveTensorArg(*stats, tag)) : std::nullopt,
      params.useNoc1Only);
}

DistributedRMSNormOpResult callDistributedRMSNorm(
    CallType callType, const ::tt::target::ttnn::DistributedRMSNormOpT &op,
    TensorArg input, std::optional<TensorArg> residual_input_tensor,
    std::optional<TensorArg> weight, std::optional<TensorArg> stats,
    const ::ttnn::GlobalSemaphore &semaphore, ::ttnn::MeshDevice *device) {
  DistributedRMSNormResolvedParams params = resolveDistributedRMSNormParams(op);

  auto makeTuple = [&](auto tag) {
    return createDistributedRMSNormTuple(tag, op, input, residual_input_tensor,
                                         weight, stats, device, semaphore,
                                         params);
  };

  return callOp<DistributedRMSNormOpResult, false, false>(
      WRAP_OP(::ttnn::fused_rms_minimal), callType, makeTuple, device,
      "DistributedRMSNormOp");
}

} // namespace ttnn_op_invoke
