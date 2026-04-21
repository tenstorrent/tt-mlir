// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFIED_OP_LIB_ELTWISE_QUANTIZATION_OP_H
#define UNIFIED_OP_LIB_ELTWISE_QUANTIZATION_OP_H

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <cstdint>
#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

template <typename T>
using TensorVariantArg =
    std::variant<std::variant<::ttnn::Tensor, T>, ::ttnn::TensorSpec>;

using EltwiseQuantizationOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseQuantizationResolvedParams {
  std::optional<int32_t> axis = std::nullopt;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDataType;
};

EltwiseQuantizationResolvedParams resolveEltwiseQuantizationParams(
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    CallType callType);

EltwiseQuantizationOpResult callEltwiseQuantizeDequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg input, TensorVariantArg<float> scale,
    TensorVariantArg<int32_t> zeroPoint, ::ttnn::MeshDevice *device = nullptr,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt);

EltwiseQuantizationOpResult callEltwiseRequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOpT,
    TensorArg input, TensorVariantArg<float> in_scale,
    TensorVariantArg<int32_t> in_zero_point, TensorVariantArg<float> out_scale,
    TensorVariantArg<int32_t> out_zero_point,
    ::ttnn::MeshDevice *device = nullptr,
    std::optional<::tt::tt_metal::DataType> outputDType = std::nullopt);

} // namespace unifiedOpLib

#endif // UNIFIED_OP_LIB_ELTWISE_QUANTIZATION_OP_H
