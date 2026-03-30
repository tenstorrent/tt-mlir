// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_CONV2D_OP_H
#define TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_CONV2D_OP_H

#include "ttmlir/Target/TTNN/operations/Conv2dParams.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/operations/conv_generated.h"
#pragma clang diagnostic pop
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/types.hpp"

#include <optional>

namespace unifiedOpLib {

using TensorArg = ::tt::target::ttnn::conv2d::TensorArg;
using Conv2dResolvedParams = ::tt::target::ttnn::conv2d::Conv2dResolvedParams;
using Conv2dOpResult = ::tt::target::ttnn::conv2d::Conv2dOpResult;

enum class CallType {
  QUERY_OP_CONSTRAINTS,
  QUERY_OP_RUNTIME,
  EXECUTE,
};

Conv2dResolvedParams
resolveConv2dParams(const ::tt::target::ttnn::Conv2dOpT &conv2dOpT);

Conv2dOpResult
callConv2d(CallType callType, const ::tt::target::ttnn::Conv2dOpT &conv2dOpT,
           TensorArg input, TensorArg weight,
           std::optional<TensorArg> bias,
           ::ttnn::MeshDevice &targetDevice);

} // namespace unifiedOpLib

#endif // TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UNIFIED_CONV2D_OP_H
