// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPINVOKE_TTNN_CONV_PREPARECONV2DBIASOP_H
#define TTMLIR_OPINVOKE_TTNN_CONV_PREPARECONV2DBIASOP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"

#include <optional>

namespace ttnn_op_invoke {

using PrepareConv2dBiasOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PrepareConv2dBiasResolvedParams {
  ::ttnn::MemoryConfig inputMemoryConfig;
  ::ttnn::Layout inputLayout;
  std::array<uint32_t, 2> kernelSize;
  std::array<uint32_t, 2> stride;
  std::array<uint32_t, 2> dilation;
  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  ::ttnn::DataType inputDtype;
  std::optional<::ttnn::DataType> outputDtype;
  std::optional<::ttnn::Conv2dConfig> conv2dConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
};

PrepareConv2dBiasResolvedParams resolvePrepareConv2dBiasParams(
    const ::tt::target::ttnn::PrepareConv2dBiasOpT &op);

PrepareConv2dBiasOpResult
callPrepareConv2dBias(CallType callType,
                      const ::tt::target::ttnn::PrepareConv2dBiasOpT &op,
                      TensorArg biasTensor, ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTMLIR_OPINVOKE_TTNN_CONV_PREPARECONV2DBIASOP_H
