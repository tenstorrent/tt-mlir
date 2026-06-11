// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_SORT_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_SORT_OP_H

#include "operations/data_movement/sort/sort.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <vector>

namespace ttnn_op_invoke {

using SortOpResult = std::variant<::ttnn::graph::ConstraintQueryResponse,
                                  ::ttnn::graph::RuntimeQueryResponse,
                                  std::vector<::ttnn::Tensor>>;

struct SortResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

SortResolvedParams resolveSortParams(const ::tt::target::ttnn::SortOpT &sortOp);

SortOpResult callSort(CallType callType,
                      const ::tt::target::ttnn::SortOpT &sortOp,
                      TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_SORT_OP_H
