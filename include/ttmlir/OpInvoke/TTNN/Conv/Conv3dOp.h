// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_CONV_3D_OP_H
#define TTNN_OP_INVOKE_CONV_3D_OP_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_query_op_runtime.hpp"
#include "ttnn/operations/experimental/conv3d/conv3d.hpp"
#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation.hpp"
#include "ttnn/types.hpp"

#include <optional>
#include <string>

namespace ttnn_op_invoke {

using Conv3dOpResult =
    std::variant<::ttnn::graph::ConstraintQueryResponse,
                 ::ttnn::graph::RuntimeQueryResponse, ::ttnn::Tensor>;

struct Conv3dResolvedParams {
  std::array<uint32_t, 3> kernelSize;
  std::array<uint32_t, 3> stride;
  std::array<uint32_t, 3> padding;
  ::ttnn::DataType outputDtype;
  std::optional<::ttnn::experimental::prim::Conv3dConfig> conv3dConfig;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig;
};

Conv3dResolvedParams
resolveConv3dParams(const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                    ::ttnn::MeshDevice *device);

Conv3dOpResult callConv3d(CallType callType,
                          const ::tt::target::ttnn::Conv3dOpT &conv3dOpT,
                          TensorArg input, TensorArg weight,
                          std::optional<TensorArg> bias,
                          ::ttnn::MeshDevice *device);

} // namespace ttnn_op_invoke

#endif // TTNN_OP_INVOKE_CONV_3D_OP_H
