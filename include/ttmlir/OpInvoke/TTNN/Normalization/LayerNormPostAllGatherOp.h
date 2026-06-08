// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_LAYER_NORM_POST_ALL_GATHER_OP_H
#define TTNN_OP_INVOKE_LAYER_NORM_POST_ALL_GATHER_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/layernorm_post_all_gather.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using LayerNormPostAllGatherOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct LayerNormPostAllGatherResolvedParams {
  std::optional<::ttnn::DataType> outputDtype;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::prim::LayerNormProgramConfig> programConfig;
};

LayerNormPostAllGatherResolvedParams resolveLayerNormPostAllGatherParams(
    const ::tt::target::ttnn::LayerNormPostAllGatherOpT &op);

LayerNormPostAllGatherOpResult callLayerNormPostAllGather(
    CallType callType, const ::tt::target::ttnn::LayerNormPostAllGatherOpT &op,
    TensorArg input, TensorArg stats, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_LAYER_NORM_POST_ALL_GATHER_OP_H
