// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_SOFTMAX_OP_H
#define TTNN_OP_INVOKE_SOFTMAX_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using SoftmaxOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct SoftmaxResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
};

SoftmaxResolvedParams
resolveSoftmaxParams(const ::tt::target::ttnn::SoftmaxOpT &opT);

SoftmaxOpResult callSoftmax(CallType callType,
                             const ::tt::target::ttnn::SoftmaxOpT &opT,
                             TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_SOFTMAX_OP_H
