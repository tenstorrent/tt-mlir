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

} // namespace unifiedOpLib
