// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_CONV_TRANSPOSE2D_OP_H
#define TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_CONV_TRANSPOSE2D_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using ConvTranspose2dOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse,
                 ::ttnn::ConvTranspose2dResultWithOptions>;

struct ConvTranspose2dResolvedParams {
  std::array<uint32_t, 2> kernelSize;
  std::array<uint32_t, 2> stride;
  std::array<uint32_t, 2> dilation;
  std::array<uint32_t, 2> padding;
  std::array<uint32_t, 2> outputPadding;
  std::optional<::ttnn::DataType> outputDtype;
  std::optional<::ttnn::Conv2dConfig> conv2dConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  std::optional<::ttnn::MemoryConfig> memoryConfig;
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
};

ConvTranspose2dResolvedParams
resolveConvTranspose2dParams(const ::tt::target::ttnn::ConvTranspose2dOpT &opT,
                             CallType callType);

ConvTranspose2dOpResult callConvTranspose2d(
    CallType callType, const ::tt::target::ttnn::ConvTranspose2dOpT &opT,
    TensorArg input, TensorArg weight, std::optional<TensorArg> bias,
    ::ttnn::MeshDevice &targetDevice);

} // namespace unifiedOpLib

#endif // TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_CONV_TRANSPOSE2D_OP_H
