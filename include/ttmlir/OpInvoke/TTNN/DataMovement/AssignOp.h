// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_ASSIGN_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_ASSIGN_OP_H

#include "operations/data_movement/copy/copy.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using AssignOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct AssignResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDtype;
};

AssignResolvedParams
resolveAssignParams(const ::tt::target::ttnn::AssignOpT &assignOp);

AssignOpResult callAssign(CallType callType,
                          const ::tt::target::ttnn::AssignOpT &assignOp,
                          TensorArg input,
                          ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_ASSIGN_OP_H
