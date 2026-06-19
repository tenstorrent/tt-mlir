// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_POOL_POOL2DOP_H
#define TTMLIR_OPINVOKE_TTNN_POOL_POOL2DOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"

#include <array>
#include <optional>
#include <variant>

namespace ttnn_op_invoke {

using AvgPool2dOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

using MaxPool2dOpResult = std::variant<::ttnn::graph::ConstraintQueryResponse,
                                       ::ttnn::graph::RuntimeQueryResponse,
                                       std::vector<::ttnn::Tensor>>;

using MaxPool2dWithIndicesOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse,
                 std::vector<::ttnn::Tensor>>;

struct Pool2dResolvedParams {
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::TensorMemoryLayout> appliedShardScheme;
  std::array<uint32_t, 2> kernelSize;
  std::array<uint32_t, 2> stride;
  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  std::optional<std::array<uint32_t, 2>> dilation;
  std::optional<bool> countIncludePad;
  ::ttnn::DataType dtype = ::ttnn::DataType::BFLOAT16;
  ::ttnn::Layout output_layout = ::ttnn::Layout::ROW_MAJOR;
};

Pool2dResolvedParams
resolvePool2dParams(const ::tt::target::ttnn::Pool2dOpT &op);

Pool2dResolvedParams resolveMaxPool2dWithIndicesParams(
    const ::tt::target::ttnn::MaxPool2dWithIndicesOpT &op);

AvgPool2dOpResult callAvgPool2d(CallType callType,
                                const ::tt::target::ttnn::Pool2dOpT &op,
                                TensorArg input, ::ttnn::MeshDevice *device);

MaxPool2dOpResult callMaxPool2d(CallType callType,
                                const ::tt::target::ttnn::Pool2dOpT &op,
                                TensorArg input, ::ttnn::MeshDevice *device);

MaxPool2dWithIndicesOpResult
callMaxPool2dWithIndices(CallType callType,
                         const ::tt::target::ttnn::MaxPool2dWithIndicesOpT &op,
                         TensorArg input, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_POOL_POOL2DOP_H
