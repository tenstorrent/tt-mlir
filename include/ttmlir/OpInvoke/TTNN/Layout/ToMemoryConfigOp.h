// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_LAYOUT_TOMEMORYCONFIGOP_H
#define TTNN_OP_INVOKE_LAYOUT_TOMEMORYCONFIGOP_H

#include "operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn_op_invoke {

using ToMemoryConfigOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ToMemoryConfigResolvedParams {
  ::ttnn::MemoryConfig outputMemoryConfig;
};

ToMemoryConfigResolvedParams
resolveToMemoryConfigParams(const ::tt::target::ttnn::ToMemoryConfigOpT &op);

ToMemoryConfigOpResult
callToMemoryConfig(CallType callType,
                   const ::tt::target::ttnn::ToMemoryConfigOpT &op,
                   TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_LAYOUT_TOMEMORYCONFIGOP_H
