// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_NORMALIZATION_DISTRIBUTEDRMSNORMOP_H
#define TTMLIR_OPINVOKE_TTNN_NORMALIZATION_DISTRIBUTEDRMSNORMOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/ccl/rms_allgather/rms_allgather.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using DistributedRMSNormOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct DistributedRMSNormResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  ::ttnn::prim::LayerNormProgramConfig programConfig;
  std::optional<::tt::tt_metal::SubDeviceId> subDeviceId;
  std::optional<size_t> numLinks;
  ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear;
  std::optional<ttnn::DataType> dtype;
  bool useNoc1Only = false;
};

DistributedRMSNormResolvedParams resolveDistributedRMSNormParams(
    const ::tt::target::ttnn::DistributedRMSNormOpT &op);

DistributedRMSNormOpResult callDistributedRMSNorm(
    CallType callType, const ::tt::target::ttnn::DistributedRMSNormOpT &op,
    TensorArg input, std::optional<TensorArg> residual_input_tensor,
    std::optional<TensorArg> weight, std::optional<TensorArg> stats,
    const ::ttnn::GlobalSemaphore &semaphore, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_NORMALIZATION_DISTRIBUTEDRMSNORMOP_H
