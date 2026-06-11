// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_PERMUTE_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_PERMUTE_OP_H

#include "operations/data_movement/permute/permute.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using PermuteOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PermuteResolvedParams {
  ::ttsl::SmallVector<int64_t> dims;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

PermuteResolvedParams
resolvePermuteParams(const ::tt::target::ttnn::PermuteOpT &permuteOp);

PermuteOpResult callPermute(CallType callType,
                            const ::tt::target::ttnn::PermuteOpT &permuteOp,
                            TensorArg input,
                            ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_PERMUTE_OP_H
