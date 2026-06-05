// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_RMS_NORM_PRE_ALL_GATHER_OP_H
#define TTNN_OP_INVOKE_RMS_NORM_PRE_ALL_GATHER_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using RMSNormPreAllGatherOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct RMSNormPreAllGatherResolvedParams {
  ::ttnn::DataType dtype;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::prim::LayerNormProgramConfig> programConfig;
  std::optional<bool> use2DCoreGrid;
};

RMSNormPreAllGatherResolvedParams resolveRMSNormPreAllGatherParams(
    const ::tt::target::ttnn::RMSNormPreAllGatherOpT &opT);

RMSNormPreAllGatherOpResult
callRMSNormPreAllGather(CallType callType,
                        const ::tt::target::ttnn::RMSNormPreAllGatherOpT &opT,
                        TensorArg input, std::optional<TensorArg> residual,
                        ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_RMS_NORM_PRE_ALL_GATHER_OP_H
