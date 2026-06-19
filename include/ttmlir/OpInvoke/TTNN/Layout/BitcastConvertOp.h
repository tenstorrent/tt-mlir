// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_LAYOUT_BITCASTCONVERTOP_H
#define TTNN_OP_INVOKE_LAYOUT_BITCASTCONVERTOP_H

#include "operations/eltwise/unary/unary.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using BitcastConvertOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct BitcastConvertResolvedParams {
  ::ttnn::DataType dtype;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

BitcastConvertResolvedParams
resolveBitcastConvertParams(const ::tt::target::ttnn::BitcastConvertOpT &op);

BitcastConvertOpResult
callBitcastConvert(CallType callType,
                   const ::tt::target::ttnn::BitcastConvertOpT &op,
                   TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_LAYOUT_BITCASTCONVERTOP_H
