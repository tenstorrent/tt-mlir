// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_SCATTER_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_SCATTER_OP_H

#include "operations/data_movement/scatter/scatter.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <string>

namespace ttnn_op_invoke {

using ScatterOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ScatterResolvedParams {
  std::optional<std::string> optReductionString;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ScatterResolvedParams
resolveScatterParams(const ::tt::target::ttnn::ScatterOpT &scatterOp);

ScatterOpResult callScatter(CallType callType,
                            const ::tt::target::ttnn::ScatterOpT &scatterOp,
                            TensorArg input, TensorArg index, TensorArg source,
                            ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_SCATTER_OP_H
