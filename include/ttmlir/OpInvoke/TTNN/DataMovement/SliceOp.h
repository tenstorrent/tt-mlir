// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_DATAMOVEMENT_SLICEOP_H
#define TTNN_OP_INVOKE_DATAMOVEMENT_SLICEOP_H

#include "operations/data_movement/slice/slice.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using SliceOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct SliceStaticResolvedParams {
  ::ttsl::SmallVector<int32_t> begins;
  ::ttsl::SmallVector<int32_t> ends;
  ::ttsl::SmallVector<int32_t> step;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

SliceStaticResolvedParams
resolveSliceStaticParams(const ::tt::target::ttnn::SliceOpT &sliceOp);

SliceOpResult callSliceStatic(CallType callType,
                              const ::tt::target::ttnn::SliceOpT &sliceOp,
                              TensorArg input,
                              ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_DATAMOVEMENT_SLICEOP_H
