// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_UNARY_COMPOSITE_OP_H
#define UNIFIED_OP_LIB_ELTWISE_UNARY_COMPOSITE_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <cstdint>
#include <optional>
#include <variant>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using EltwiseUnaryCompositeOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseUnaryCompositeClampScalarResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::variant<float, int32_t> min;
  std::variant<float, int32_t> max;
};

struct EltwiseUnaryCompositeClampTensorResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

EltwiseUnaryCompositeClampScalarResolvedParams
resolveEltwiseUnaryCompositeClampScalarParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT);

EltwiseUnaryCompositeClampTensorResolvedParams
resolveEltwiseUnaryCompositeClampTensorParams(
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT);

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampScalar(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt);

EltwiseUnaryCompositeOpResult callEltwiseUnaryCompositeClampTensor(
    CallType callType,
    const ::tt::target::ttnn::EltwiseUnaryCompositeOpT
        &eltwiseUnaryCompositeOpT,
    TensorArg input, TensorArg min, TensorArg max,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::ttnn::MemoryConfig> outputMemoryConfig = std::nullopt);

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_UNARY_COMPOSITE_OP_H
