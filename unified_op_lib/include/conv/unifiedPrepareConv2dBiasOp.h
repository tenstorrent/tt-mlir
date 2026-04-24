// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_PREPARE_CONV2D_BIAS_OP_H
#define TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_PREPARE_CONV2D_BIAS_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/types.hpp"
#include "utils/utils.h"

#include <optional>

namespace unifiedOpLib {

using TensorArg = std::variant<const ::ttnn::Tensor *, ::ttnn::TensorSpec>;

using PrepareConv2dBiasOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct PrepareConv2dBiasResolvedParams {
  std::array<uint32_t, 2> kernelSize;
  std::array<uint32_t, 2> stride;
  std::array<uint32_t, 2> dilation;
  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  ::ttnn::DataType inputDtype;
  std::optional<::ttnn::DataType> outputDtype;
  ::ttnn::MemoryConfig inputMemoryConfig;
  ::ttnn::Layout inputLayout;
  std::optional<::ttnn::Conv2dConfig> conv2dConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
};

PrepareConv2dBiasResolvedParams resolvePrepareConv2dBiasParams(
    const ::tt::target::ttnn::PrepareConv2dBiasOpT &opT, CallType callType);

PrepareConv2dBiasOpResult
callPrepareConv2dBias(CallType callType,
                      const ::tt::target::ttnn::PrepareConv2dBiasOpT &opT,
                      TensorArg bias, ::ttnn::MeshDevice &targetDevice);

} // namespace unifiedOpLib

#endif // TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_PREPARE_CONV2D_BIAS_OP_H
