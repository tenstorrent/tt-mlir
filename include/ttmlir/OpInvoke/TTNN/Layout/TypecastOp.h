// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_LAYOUT_TYPECASTOP_H
#define TTNN_OP_INVOKE_LAYOUT_TYPECASTOP_H

#include "operations/copy/typecast/typecast.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using TypecastOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct TypecastResolvedParams {
  ::ttnn::DataType dtype;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

TypecastResolvedParams
resolveTypecastParams(const ::tt::target::ttnn::TypecastOpT &op);

TypecastOpResult callTypecast(CallType callType,
                              const ::tt::target::ttnn::TypecastOpT &op,
                              TensorArg input,
                              ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_LAYOUT_TYPECASTOP_H
