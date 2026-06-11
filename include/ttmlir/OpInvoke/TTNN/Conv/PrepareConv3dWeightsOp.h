// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_CONV_PREPARECONV3DWEIGHTSOP_H
#define TTMLIR_OPINVOKE_TTNN_CONV_PREPARECONV3DWEIGHTSOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp"

namespace ttnn_op_invoke {

using PrepareConv3dWeightsOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PrepareConv3dWeightsResolvedParams {};

PrepareConv3dWeightsResolvedParams resolvePrepareConv3dWeightsParams(
    const ::tt::target::ttnn::PrepareConv3dWeightsOpT &op);

PrepareConv3dWeightsOpResult
callPrepareConv3dWeights(CallType callType,
                         const ::tt::target::ttnn::PrepareConv3dWeightsOpT &op,
                         TensorArg weightTensor, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_CONV_PREPARECONV3DWEIGHTSOP_H
