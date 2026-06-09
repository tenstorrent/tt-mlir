// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_CONCAT_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_CONCAT_OP_H

#include "operations/data_movement/concat/concat.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

#include <optional>
#include <vector>

namespace ttnn_op_invoke {

using ConcatOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ConcatResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ConcatResolvedParams
resolveConcatParams(const ::tt::target::ttnn::ConcatOpT &concatOp);

ConcatOpResult callConcat(CallType callType,
                          const ::tt::target::ttnn::ConcatOpT &concatOp,
                          std::vector<TensorArg> inputs,
                          ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_CONCAT_OP_H
