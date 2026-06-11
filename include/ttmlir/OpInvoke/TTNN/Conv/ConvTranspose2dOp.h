// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_CONV_CONVTRANSPOSE2DOP_H
#define TTMLIR_OPINVOKE_TTNN_CONV_CONVTRANSPOSE2DOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"

#include <optional>

namespace ttnn_op_invoke {

using ConvTranspose2dOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse,
                 ::ttnn::ConvTranspose2dResultWithOptions>;

struct ConvTranspose2dResolvedParams {
  std::array<uint32_t, 2> kernelSize;
  std::array<uint32_t, 2> stride;
  std::array<uint32_t, 2> padding;
  std::array<uint32_t, 2> outputPadding;
  std::array<uint32_t, 2> dilation;
  std::optional<::ttnn::DataType> outputDtype;
  std::optional<::ttnn::Conv2dConfig> conv2dConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
};

ConvTranspose2dResolvedParams resolveConvTranspose2dParams(
    const ::tt::target::ttnn::ConvTranspose2dOpT &convTranspose2dOpT);

ConvTranspose2dOpResult callConvTranspose2d(
    CallType callType,
    const ::tt::target::ttnn::ConvTranspose2dOpT &convTranspose2dOpT,
    TensorArg input, TensorArg weight, std::optional<TensorArg> bias,
    ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_CONV_CONVTRANSPOSE2DOP_H
