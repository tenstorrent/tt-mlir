// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_GATHER_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_GATHER_OP_H

#include "operations/data_movement/gather/gather.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using GatherOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct GatherResolvedParams {
  int8_t dim;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

GatherResolvedParams
resolveGatherParams(const ::tt::target::ttnn::GatherOpT &gatherOp);

GatherOpResult callGather(CallType callType,
                          const ::tt::target::ttnn::GatherOpT &gatherOp,
                          TensorArg input, TensorArg index,
                          ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_GATHER_OP_H
