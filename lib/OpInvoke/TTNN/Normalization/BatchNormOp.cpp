// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Normalization/BatchNormOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace ttnn_op_invoke {

BatchNormResolvedParams resolveBatchNormInferenceParams(
    const ::tt::target::ttnn::BatchNormInferenceOpT &opT) {
  BatchNormResolvedParams params;
  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
  }
  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  return params;
}

BatchNormResolvedParams resolveBatchNormTrainingParams(
    const ::tt::target::ttnn::BatchNormTrainingOpT &opT) {
  BatchNormResolvedParams params;
  if (opT.compute_config) {
    params.computeConfig =
        operations::utils::createDeviceComputeKernelConfig(*opT.compute_config);
  }
  if (opT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*opT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*opT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }
  return params;
}

template <typename Tag>
auto createBatchNormInferenceTuple(
    Tag tag, const ::tt::target::ttnn::BatchNormInferenceOpT &opT,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, const BatchNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag),
      runningMean ? std::make_optional(resolveTensorArg(*runningMean, tag))
                  : std::nullopt,
      runningVar ? std::make_optional(resolveTensorArg(*runningVar, tag))
                 : std::nullopt,
      /*training=*/false, opT.epsilon, /*momentum=*/0.1f,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      std::optional<::ttnn::Tensor>(std::nullopt), params.outputMemoryConfig,
      params.computeConfig);
}

template <typename Tag>
auto createBatchNormTrainingTuple(
    Tag tag, const ::tt::target::ttnn::BatchNormTrainingOpT &opT,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, const BatchNormResolvedParams &params) {
  return std::make_tuple(
      resolveTensorArg(input, tag),
      runningMean ? std::make_optional(resolveTensorArg(*runningMean, tag))
                  : std::nullopt,
      runningVar ? std::make_optional(resolveTensorArg(*runningVar, tag))
                 : std::nullopt,
      /*training=*/true, opT.epsilon, opT.momentum,
      weight ? std::make_optional(resolveTensorArg(*weight, tag))
             : std::nullopt,
      bias ? std::make_optional(resolveTensorArg(*bias, tag)) : std::nullopt,
      std::optional<::ttnn::Tensor>(std::nullopt), params.outputMemoryConfig,
      params.computeConfig);
}

BatchNormOpResult callBatchNormInference(
    CallType callType, const ::tt::target::ttnn::BatchNormInferenceOpT &opT,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  BatchNormResolvedParams params =
      resolveBatchNormInferenceParams(opT);

  auto makeTuple = [&](auto tag) {
    return createBatchNormInferenceTuple(tag, opT, input, runningMean,
                                         runningVar, weight, bias, params);
  };

  callOp(::ttnn::batch_norm);
}

BatchNormOpResult callBatchNormTraining(
    CallType callType, const ::tt::target::ttnn::BatchNormTrainingOpT &opT,
    TensorArg input, std::optional<TensorArg> runningMean,
    std::optional<TensorArg> runningVar, std::optional<TensorArg> weight,
    std::optional<TensorArg> bias, ::ttnn::MeshDevice *device) {
  BatchNormResolvedParams params =
      resolveBatchNormTrainingParams(opT);

  auto makeTuple = [&](auto tag) {
    return createBatchNormTrainingTuple(tag, opT, input, runningMean,
                                        runningVar, weight, bias, params);
  };

  callOp(::ttnn::batch_norm);
}

} // namespace ttnn_op_invoke
