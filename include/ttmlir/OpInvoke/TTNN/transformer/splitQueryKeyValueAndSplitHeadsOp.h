// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_SPLIT_QUERY_KEY_VALUE_AND_SPLIT_HEADS_OP_H
#define TTNN_OP_INVOKE_SPLIT_QUERY_KEY_VALUE_AND_SPLIT_HEADS_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/transformer_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"
#include "ttnn/types.hpp"

#include <optional>
#include <tuple>

namespace ttnn_op_invoke {

using SplitQueryKeyValueAndSplitHeadsOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse,
                 std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor>>;

struct SplitQueryKeyValueAndSplitHeadsResolvedParams {
  std::optional<uint32_t> numKVHeads;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

SplitQueryKeyValueAndSplitHeadsResolvedParams
resolveSplitQueryKeyValueAndSplitHeadsParams(
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opT,
    CallType callType);

SplitQueryKeyValueAndSplitHeadsOpResult callSplitQueryKeyValueAndSplitHeads(
    CallType callType,
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT &opT,
    TensorArg input, std::optional<TensorArg> kvInput,
    ::ttnn::MeshDevice &targetDevice);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_SPLIT_QUERY_KEY_VALUE_AND_SPLIT_HEADS_OP_H
