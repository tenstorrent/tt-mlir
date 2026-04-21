// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                    CallType callType) {

  MatmulResolvedParams params;

  if (matmulOpT.out) {
    params.outputDType = operations::utils::getDataType(*matmulOpT.out);
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
        operations::utils::getTensorRefMemoryConfig(*matmulOpT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*matmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

MatmulOpResult callMatmul(CallType callType,
                          const ::tt::target::ttnn::MatmulOpT &matmulOpT,
                          TensorArg lhs, TensorArg rhs,
                          ::ttnn::MeshDevice *device) {

  MatmulResolvedParams params = resolveMatmulParams(matmulOpT, callType);

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
                          params.outputMemoryConfig, params.outputDType,
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
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOpT,
                    CallType callType) {

  LinearResolvedParams params;

  if (linearOpT.out) {
    params.outputDType = operations::utils::getDataType(*linearOpT.out);
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
        operations::utils::getTensorRefMemoryConfig(*linearOpT.out), callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*linearOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

LinearOpResult callLinear(CallType callType,
                          const ::tt::target::ttnn::LinearOpT &linearOpT,
                          TensorArg a, TensorArg b,
                          const std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device) {
  LinearResolvedParams params = resolveLinearParams(linearOpT, callType);

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
                          params.outputDType, params.matmulProgramConfig,
                          /*activation=*/params.activation,
                          /*compute_kernel_config=*/params.computeConfig,
                          /*core_grid=*/std::nullopt,
                          /*output_tile=*/std::nullopt,
                          /* optional_output_tensor=*/std::nullopt);
  }
  }
}

SparseMatmulResolvedParams resolveSparseMatmulParams(
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
    CallType callType) {

  SparseMatmulResolvedParams params;

  if (sparseMatmulOpT.nnz != 0) {
    params.nnz = std::make_optional(static_cast<uint32_t>(sparseMatmulOpT.nnz));
  }

  auto matmulProgramConfig =
      operations::utils::createMatmulProgramConfigIfNeeded(sparseMatmulOpT);
  LOG_ASSERT(matmulProgramConfig.has_value(),
             "SparseMatmulOp requires program_config to be set at compile "
             "time");
  params.matmulProgramConfig = matmulProgramConfig.value();

  if (sparseMatmulOpT.compute_config) {
    params.computeConfig = operations::utils::createDeviceComputeKernelConfig(
        *sparseMatmulOpT.compute_config);
  }

  if (sparseMatmulOpT.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*sparseMatmulOpT.out),
        callType);
    LOG_ASSERT(operations::utils::inSystemMemory(*sparseMatmulOpT.out) ||
                   params.outputMemoryConfig.has_value(),
               "Memory config must exist for device tensors");
  }

  return params;
}

SparseMatmulOpResult
callSparseMatmul(CallType callType,
                 const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT,
                 TensorArg a, TensorArg b, TensorArg sparsity,
                 ::ttnn::MeshDevice *device) {
  SparseMatmulResolvedParams params =
      resolveSparseMatmulParams(sparseMatmulOpT, callType);

  switch (callType) {
  case CallType::QUERY_OP_CONSTRAINTS: {
    ::ttnn::graph::ConstraintQueryResponse response;
    response.error_message =
        "Constraint query not implemented for SparseMatmulOp yet";
    return response;
  }
  case CallType::QUERY_OP_RUNTIME: {
    ::ttnn::graph::RuntimeQueryResponse response;
    response.error_message =
        "Runtime query not implemented for SparseMatmulOp yet";
    return response;
  }
  case CallType::EXECUTE: {
    const auto &input_a = *std::get<const ::ttnn::Tensor *>(a);
    const auto &input_b = *std::get<const ::ttnn::Tensor *>(b);
    const auto &input_sparsity = *std::get<const ::ttnn::Tensor *>(sparsity);

    return ::ttnn::sparse_matmul(
        input_a, input_b, input_sparsity,
        /*program_config=*/params.matmulProgramConfig,
        /*nnz=*/params.nnz,
        /*is_input_a_sparse=*/sparseMatmulOpT.is_input_a_sparse,
        /*is_input_b_sparse=*/sparseMatmulOpT.is_input_b_sparse,
        /*memory_config=*/params.outputMemoryConfig,
        /*dtype=*/std::nullopt,
        /*compute_kernel_config=*/params.computeConfig,
        /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt);
  }
  }
}

} // namespace unifiedOpLib
