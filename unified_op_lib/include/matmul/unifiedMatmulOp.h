// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_MATMUL_OP_H
#define UNIFIED_OP_LIB_MATMUL_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic pop
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

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_MATMUL_OP_H
