// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_MATMUL_MATMULOP_H
#define TTMLIR_OPINVOKE_TTNN_MATMUL_MATMULOP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"

#include <optional>

namespace ttnn_op_invoke {

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
  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulProgramConfig;
  std::optional<std::string> activation;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
};

MatmulResolvedParams
resolveMatmulParams(const ::tt::target::ttnn::MatmulOpT &matmulOp);

MatmulOpResult callMatmul(CallType callType,
                          const ::tt::target::ttnn::MatmulOpT &matmulOp,
                          TensorArg lhs, TensorArg rhs,
                          ::ttnn::MeshDevice *device = nullptr);

struct LinearResolvedParams {
  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulProgramConfig;
  std::optional<std::string> activation;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
};

LinearResolvedParams
resolveLinearParams(const ::tt::target::ttnn::LinearOpT &linearOp);

LinearOpResult callLinear(CallType callType,
                          const ::tt::target::ttnn::LinearOpT &linearOp,
                          TensorArg lhs, TensorArg rhs,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device = nullptr);

struct SparseMatmulResolvedParams {
  ::ttnn::operations::matmul::MatmulProgramConfig matmulProgramConfig;
  std::optional<uint32_t> nnz;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDType;
};

SparseMatmulResolvedParams resolveSparseMatmulParams(
    const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOp);

SparseMatmulOpResult
callSparseMatmul(CallType callType,
                 const ::tt::target::ttnn::SparseMatmulOpT &sparseMatmulOp,
                 TensorArg a, TensorArg b, TensorArg sparsity,
                 ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_MATMUL_MATMULOP_H
