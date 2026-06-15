// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_REDUCTION_PRODOP_H
#define TTMLIR_OPINVOKE_TTNN_REDUCTION_PRODOP_H

#include "operations/reduction/prod/prod.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ProdOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ProdResolvedParams {
  std::optional<int64_t> dimArg;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

ProdResolvedParams
resolveProdParams(const ::tt::target::ttnn::ReductionProdOpT &prodOp);

ProdOpResult callProd(CallType callType,
                      const ::tt::target::ttnn::ReductionProdOpT &prodOp,
                      TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_REDUCTION_PRODOP_H
