// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_ELTWISE_QUANTIZATION_ELTWISEQUANTIZATIONOP_H
#define TTMLIR_OPINVOKE_TTNN_ELTWISE_QUANTIZATION_ELTWISEQUANTIZATIONOP_H

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"

#include <cstdint>
#include <optional>

namespace ttnn_op_invoke {

template <typename T>
using TensorVariantArg =
    std::variant<std::variant<::ttnn::Tensor, T>, ::ttnn::TensorSpec>;

template <typename T>
inline auto resolveTensorVariantArg(TensorVariantArg<T> arg,
                                    QueryTag callType) {
  return std::get<::ttnn::TensorSpec>(arg);
}

template <typename T>
inline auto resolveTensorVariantArg(TensorVariantArg<T> arg,
                                    ExecuteTag callType) {
  return std::get<std::variant<::ttnn::Tensor, T>>(arg);
}

using EltwiseQuantizationOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct EltwiseQuantizationResolvedParams {
  std::optional<int32_t> axis = std::nullopt;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::tt::tt_metal::DataType> outputDataType;
};

EltwiseQuantizationResolvedParams resolveEltwiseQuantizationParams(
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp);

EltwiseQuantizationOpResult callEltwiseQuantizeDequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp,
    TensorArg input, TensorVariantArg<float> scale,
    TensorVariantArg<int32_t> zeroPoint, ::ttnn::MeshDevice *device = nullptr);

EltwiseQuantizationOpResult callEltwiseRequantize(
    CallType callType,
    const ::tt::target::ttnn::EltwiseQuantizationOpT &eltwiseQuantizationOp,
    TensorArg input, TensorVariantArg<float> in_scale,
    TensorVariantArg<int32_t> in_zero_point, TensorVariantArg<float> out_scale,
    TensorVariantArg<int32_t> out_zero_point,
    ::ttnn::MeshDevice *device = nullptr);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_ELTWISE_QUANTIZATION_ELTWISEQUANTIZATIONOP_H
