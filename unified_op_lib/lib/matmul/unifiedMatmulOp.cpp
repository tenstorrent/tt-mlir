// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul/unifiedMatmulOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/matmul_generated.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "utils/utils.h"
#include <operations/conv/conv2d/conv2d.hpp>
#include <operations/functions.hpp>
#include <optional>
#include <variant>

namespace unifiedOpLib {

MatmulResolvedParams
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOpT) {

  MatmulResolvedParams params;

  if (matmulOpT.out) {
    params.outputDataType = operations::utils::getDataType(*matmulOpT.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(matmulOpT);
  params.activation = matmulOpT.activation.empty()
                          ? std::nullopt
                          : std::make_optional(matmulOpT.activation);
  if (matmulOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *matmulOpT.compute_config);
  }

  if (matmulOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*matmulOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*matmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

MatmulOpResult
callMatmul(CallType callType, const ::tt::target::ttnn::MatmulOpT &matmulOpT,
           TensorArg lhs, TensorArg rhs, ::ttnn::MeshDevice *device,
           std::optional<::ttnn::MemoryConfig> outputMemoryConfig,
           std::optional<::tt::tt_metal::DataType> outputDType) {

  MatmulResolvedParams params = resolveMatmulParams(matmulOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }
  if (outputDType.has_value()) {
    params.outputDType = outputDType;
  }

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::matmul, device, std::get<::ttnn::TensorSpec>(lhs),
        std::get<::ttnn::TensorSpec>(rhs), matmulOpT.transpose_a,
        matmulOpT.transpose_b, params.outputMemoryConfig, params.outputDType,
        params.matmulProgramConfig, params.activation, params.computeConfig,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt, /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(
        ::ttnn::matmul, device, std::get<::ttnn::TensorSpec>(lhs),
        std::get<::ttnn::TensorSpec>(rhs), matmulOpT.transpose_a,
        matmulOpT.transpose_b, params.outputMemoryConfig, params.outputDType,
        params.matmulProgramConfig, params.activation, params.computeConfig,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt, /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt,
        /*sub_device_id=*/std::nullopt);
  case CallType::EXECUTE: {
    const auto &a = *std::get<const ::ttnn::Tensor *>(lhs);
    const auto &b = *std::get<const ::ttnn::Tensor *>(rhs);
    return ::ttnn::matmul(a, b, matmulOpT.transpose_a, matmulOpT.transpose_b,
                          params.outputMemoryConfig, params.outputDataType,
                          params.matmulProgramConfig,
                          /*activation=*/params.activation,
                          /*compute_kernel_config=*/params.computeConfig,
                          /*core_grid=*/std::nullopt,
                          /*output_tile=*/std::nullopt,
                          /*optional_output_tensor=*/std::nullopt,
                          /*global_cb=*/std::nullopt,
                          /*sub_device_id=*/std::nullopt);
  }
  }
}

LinearResolvedParams
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOpT) {

  LinearResolvedParams params;

  if (linearOpT.out) {
    params.outputDataType = operations::utils::getDataType(*linearOpT.out);
  }
  params.matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(linearOpT);
  params.activation = linearOpT.activation.empty()
                          ? std::nullopt
                          : std::make_optional(linearOpT.activation);
  if (linearOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *linearOpT.compute_config);
  }

  if (linearOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*linearOpT.out));
    LOG_ASSERT(operations::utils::inSystemMemory(*linearOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

LinearOpResult
callLinear(CallType callType, const ::tt::target::ttnn::LinearOpT &linearOpT,
           TensorArg a, TensorArg b, const std::optional<TensorArg> bias,
           ::ttnn::MeshDevice *device,
           std::optional<::ttnn::MemoryConfig> outputMemoryConfig,
           std::optional<::tt::tt_metal::DataType> outputDType) {
  LinearResolvedParams params = resolveLinearParams(linearOpT);
  if (outputMemoryConfig.has_value()) {
    params.outputMemoryConfig = outputMemoryConfig;
  }
  if (outputDType.has_value()) {
    params.outputDType = outputDType;
  }
  const auto &biasTensor =
      bias.has_value()
          ? std::make_optional(*std::get<const ::ttnn::Tensor *>(bias.value()))
          : std::nullopt;

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS:
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::linear, device, std::get<::ttnn::TensorSpec>(a),
        std::get<::ttnn::TensorSpec>(b), biasTensor, linearOpT.transpose_a,
        linearOpT.transpose_b, params.outputMemoryConfig, params.outputDType,
        params.matmulProgramConfig, params.activation, params.computeConfig,
        /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
  case CallType::QUERY_OP_RUNTIME:
    return QUERY_OP_RUNTIME(
        ::ttnn::linear, device, std::get<::ttnn::TensorSpec>(a),
        std::get<::ttnn::TensorSpec>(b), biasTensor, linearOpT.transpose_a,
        linearOpT.transpose_b, params.outputMemoryConfig, params.outputDType,
        params.matmulProgramConfig, params.activation, params.computeConfig,
        /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
  case CallType::EXECUTE: {
    const auto &input_a = *std::get<const ::ttnn::Tensor *>(a);
    const auto &input_b = *std::get<const ::ttnn::Tensor *>(b);

    return ::ttnn::linear(input_a, input_b, biasTensor, linearOpT.transpose_a,
                          linearOpT.transpose_b, params.outputMemoryConfig,
                          params.outputDataType, params.matmulProgramConfig,
                          /*activation=*/params.activation,
                          /*compute_kernel_config=*/params.computeConfig,
                          /*core_grid=*/std::nullopt,
                          /*output_tile=*/std::nullopt,
                          /* optional_output_tensor=*/std::nullopt);
  }
  }
}

} // namespace unifiedOpLib
