// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_CONV2D_OP_H
#define TTNN_OP_INVOKE_CONV2D_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace ttnn_op_invoke {

using Conv2dOpResult = std::variant<::ttnn::graph::ConstraintQueryResponse,
                                    ::ttnn::graph::RuntimeQueryResponse,
                                    ::ttnn::Conv2dResultWithOptions>;

struct Conv2dResolvedParams {
  std::array<uint32_t, 2> kernelSize;
  std::array<uint32_t, 2> stride;
  std::array<uint32_t, 2> dilation;
  std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>> padding;
  std::optional<::ttnn::DataType> outputDtype;
  std::optional<::ttnn::Conv2dConfig> conv2dConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig;
};

Conv2dResolvedParams
resolveConv2dParams(const ::tt::target::ttnn::Conv2dOpT &conv2dOp);

Conv2dOpResult callConv2d(CallType callType,
                          const ::tt::target::ttnn::Conv2dOpT &conv2dOp,
                          TensorArg input, TensorArg weight,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *targetDevice);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_CONV2D_OP_H
