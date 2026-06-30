// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_REDUCTION_CUMSUMOP_H
#define TTMLIR_OPINVOKE_TTNN_REDUCTION_CUMSUMOP_H

#include "operations/reduction/accumulation/cumsum/cumsum.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using CumSumOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct CumSumResolvedParams {
  std::optional<::ttnn::DataType> dtype;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

CumSumResolvedParams
resolveCumSumParams(const ::tt::target::ttnn::CumSumOpT &cumSumOp);

CumSumOpResult callCumSum(CallType callType,
                          const ::tt::target::ttnn::CumSumOpT &cumSumOp,
                          TensorArg input,
                          ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_REDUCTION_CUMSUMOP_H
