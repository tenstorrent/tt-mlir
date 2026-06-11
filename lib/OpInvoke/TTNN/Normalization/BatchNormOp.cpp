// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/BatchNormOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"

#include <optional>

namespace ttnn_op_invoke {

BatchNormResolvedParams resolveBatchNormInferenceParams(
    const ::tt::target::ttnn::BatchNormInferenceOpT &op) {
  BatchNormResolvedParams params;
  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }
  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }
  return params;
}

BatchNormResolvedParams resolveBatchNormTrainingParams(
    const ::tt::target::ttnn::BatchNormTrainingOpT &op) {
  BatchNormResolvedParams params;
  if (op.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*op.compute_config);
  }
  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createBatchNormInferenceTuple(
    Tag tag, const ::tt::target::ttnn::BatchNormInferenceOpT &op,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, const BatchNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag),
      runningMean ? std::make_optional(resolveTensorArg(*runningMean, tag))
                  : std::nullopt,
      runningVar ? std::make_optional(resolveTensorArg(*runningVar, tag))
                 : std::nullopt,
      /*training=*/false, op.epsilon, /*momentum=*/0.1f,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      std::optional<::ttnn::Tensor>(std::nullopt), params.outputMemoryConfig,
      params.computeConfig);
}

template <typename Tag>
auto createBatchNormTrainingTuple(
    Tag tag, const ::tt::target::ttnn::BatchNormTrainingOpT &op,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, const BatchNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag),
      runningMean ? std::make_optional(resolveTensorArg(*runningMean, tag))
                  : std::nullopt,
      runningVar ? std::make_optional(resolveTensorArg(*runningVar, tag))
                 : std::nullopt,
      /*training=*/true, op.epsilon, op.momentum,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      std::optional<::ttnn::Tensor>(std::nullopt), params.outputMemoryConfig,
      params.computeConfig);
}

BatchNormOpResult callBatchNormInference(
    CallType callType, const ::tt::target::ttnn::BatchNormInferenceOpT &op,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  BatchNormResolvedParams params = resolveBatchNormInferenceParams(op);

  auto makeTuple = [&](auto tag) {
    return createBatchNormInferenceTuple(tag, op, input, runningMean,
                                         runningVar, weight, bias, params);
  };

  return callOp<BatchNormOpResult>(WRAP_OP(::ttnn::batch_norm), callType,
                                   makeTuple, device);
}

BatchNormOpResult callBatchNormTraining(
    CallType callType, const ::tt::target::ttnn::BatchNormTrainingOpT &op,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  BatchNormResolvedParams params = resolveBatchNormTrainingParams(op);

  auto makeTuple = [&](auto tag) {
    return createBatchNormTrainingTuple(tag, op, input, runningMean, runningVar,
                                        weight, bias, params);
  };

  return callOp<BatchNormOpResult>(WRAP_OP(::ttnn::batch_norm), callType,
                                   makeTuple, device);
}

} // namespace ttnn_op_invoke
