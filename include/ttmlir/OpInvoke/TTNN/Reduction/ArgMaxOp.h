// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_REDUCTION_ARGMAXOP_H
#define TTMLIR_OPINVOKE_TTNN_REDUCTION_ARGMAXOP_H

#include "operations/reduction/argmax/argmax.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ArgMaxOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ArgMaxResolvedParams {
  std::optional<int> dim;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ArgMaxResolvedParams
resolveArgMaxParams(const ::tt::target::ttnn::ReductionArgMaxOpT &argMaxOp);

ArgMaxOpResult
callArgMax(CallType callType,
           const ::tt::target::ttnn::ReductionArgMaxOpT &argMaxOp,
           TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_REDUCTION_ARGMAXOP_H
