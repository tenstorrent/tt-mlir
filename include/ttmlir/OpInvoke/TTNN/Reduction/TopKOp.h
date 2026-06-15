// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_REDUCTION_TOPKOP_H
#define TTMLIR_OPINVOKE_TTNN_REDUCTION_TOPKOP_H

#include "operations/reduction/topk/topk.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <vector>

namespace ttnn_op_invoke {

using TopKOpResult = std::variant<::ttnn::graph::ConstraintQueryResponse,
                                  ::ttnn::graph::RuntimeQueryResponse,
                                  std::vector<::ttnn::Tensor>>;

struct TopKResolvedParams {
  uint32_t k;
  int8_t dim;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

TopKResolvedParams resolveTopKParams(const ::tt::target::ttnn::TopKOpT &op);

TopKOpResult callTopK(CallType callType, const ::tt::target::ttnn::TopKOpT &op,
                      TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_REDUCTION_TOPKOP_H
