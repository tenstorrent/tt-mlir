// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATA_MOVEMENT_REPEAT_OP_H
#define TTNN_OP_INVOKE_DATA_MOVEMENT_REPEAT_OP_H

#include "operations/data_movement/repeat/repeat.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using RepeatOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct RepeatResolvedParams {
  ::ttsl::SmallVector<uint32_t> repeatDims;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

RepeatResolvedParams
resolveRepeatParams(const ::tt::target::ttnn::RepeatOpT &repeatOp);

RepeatOpResult callRepeat(CallType callType,
                          const ::tt::target::ttnn::RepeatOpT &repeatOp,
                          TensorArg input,
                          ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATA_MOVEMENT_REPEAT_OP_H
