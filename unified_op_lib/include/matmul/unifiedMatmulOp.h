// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_MATMUL_OP_H
#define UNIFIED_OP_LIB_MATMUL_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;
using MatmulOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;
using LinearOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;
using SparseMatmulOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct MatmulResolvedParams {
  ::ttnn::DataType outputDataType;
  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulProgramConfig;
  std::optional<std::string> activation;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
};

MatmulResolvedParams
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOpT);

MatmulOpResult callMatmul(
    CallType callType, const ::tt::target::ttnn::MatmulOpT &matmulOpT,
    TensorArg lhs, TensorArg rhs, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt);

struct LinearResolvedParams {
  ::ttnn::DataType outputDataType;
  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulProgramConfig;
  std::optional<std::string> activation;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
  std::optional<::ttnn::Tensor> biasTensor;
};

LinearResolvedParams
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOpT);

LinearOpResult callLinear(
    CallType callType, const ::tt::target::ttnn::LinearOpT &linearOpT,
    TensorArg lhs, TensorArg rhs, std::optional<TensorArg> bias,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt);

struct SparseMatmulResolvedParams {
  ::ttnn::operations::matmul::MatmulProgramConfig matmulProgramConfig;
  std::optional<uint32_t> nnz;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
  std::optional<::ttnn::Tensor> biasTensor;
};

SparseMatmulResolvedParams resolveSparseMatmulParams(
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT);

SparseMatmulOpResult callSparseMatmul(
    CallType callType,
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOpT, TensorArg a,
    TensorArg b, TensorArg sparsity, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt);

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_MATMUL_OP_H
