// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_NORMALIZATION_BATCHNORMOP_H
#define TTMLIR_OPINVOKE_TTNN_NORMALIZATION_BATCHNORMOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/normalization_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"

#include <optional>

namespace ttnn_op_invoke {

using BatchNormOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct BatchNormResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
};

BatchNormResolvedParams resolveBatchNormInferenceParams(
    const ::tt::target::ttnn::BatchNormInferenceOpT &op);

BatchNormResolvedParams resolveBatchNormTrainingParams(
    const ::tt::target::ttnn::BatchNormTrainingOpT &op);

BatchNormOpResult callBatchNormInference(
    CallType callType, const ::tt::target::ttnn::BatchNormInferenceOpT &op,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device);

BatchNormOpResult callBatchNormTraining(
    CallType callType, const ::tt::target::ttnn::BatchNormTrainingOpT &op,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_NORMALIZATION_BATCHNORMOP_H
