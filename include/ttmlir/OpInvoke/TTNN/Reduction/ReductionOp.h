// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_REDUCTION_REDUCTIONOP_H
#define TTMLIR_OPINVOKE_TTNN_REDUCTION_REDUCTIONOP_H

#include "operations/reduction/generic/generic_reductions.hpp"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ReductionOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct ReductionResolvedParams {
  std::optional<::ttsl::SmallVector<int>> dimArg;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
};

ReductionResolvedParams
resolveReductionParams(const ::tt::target::ttnn::ReductionOpT &reductionOp);

ReductionOpResult
callReduction(CallType callType,
              const ::tt::target::ttnn::ReductionOpT &reductionOp,
              TensorArg input, ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_REDUCTION_REDUCTIONOP_H
