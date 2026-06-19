// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_LAYOUT_TOLAYOUTOP_H
#define TTNN_OP_INVOKE_LAYOUT_TOLAYOUTOP_H

#include "operations/core/to_layout/to_layout_op.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ToLayoutOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ToLayoutResolvedParams {
  ::ttnn::Layout layout;
  std::optional<::ttnn::DataType> dtype;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ToLayoutResolvedParams
resolveToLayoutParams(const ::tt::target::ttnn::ToLayoutOpT &op);

ToLayoutOpResult callToLayout(CallType callType,
                              const ::tt::target::ttnn::ToLayoutOpT &op,
                              TensorArg input,
                              ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_LAYOUT_TOLAYOUTOP_H
