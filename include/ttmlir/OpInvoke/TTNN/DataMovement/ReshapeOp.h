// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_RESHAPE_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_RESHAPE_OP_H

#include "operations/data_movement/reshape_view/reshape.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ReshapeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ReshapeResolvedParams {
  std::vector<int32_t> shape;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ReshapeResolvedParams
resolveReshapeParams(const ::tt::target::ttnn::ReshapeOpT &reshapeOp);

ReshapeOpResult callReshape(CallType callType,
                            const ::tt::target::ttnn::ReshapeOpT &reshapeOp,
                            TensorArg input,
                            ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_RESHAPE_OP_H
