// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_NORMALIZATION_RMSNORMOP_H
#define TTMLIR_OPINVOKE_TTNN_NORMALIZATION_RMSNORMOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"

#include <optional>

namespace ttnn_op_invoke {

using RMSNormOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct RMSNormResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
};

RMSNormResolvedParams
resolveRMSNormParams(const ::tt::target::ttnn::RMSNormOpT &op);

RMSNormOpResult callRMSNorm(CallType callType,
                            const ::tt::target::ttnn::RMSNormOpT &op,
                            TensorArg input, std::optional<TensorArg> weight,
                            std::optional<TensorArg> bias,
                            ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_NORMALIZATION_RMSNORMOP_H
