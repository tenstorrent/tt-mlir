// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_OP_INVOKE_UTILS_H
#define TTNN_OP_INVOKE_UTILS_H

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/TTNN/program_generated.h"
#pragma clang diagnostic pop
#include "ttmlir/OpModel/TTNN/MetalHeaders.h"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

// Macros to wrap overloaded functions for use with
// query_op_constraints/runtime. These create a generic lambda that forwards
// arguments, letting the compiler resolve the correct overload based on the
// actual argument types.
// clang-format off
#define WRAP_OP(op)                                                            \
  [&](auto &&...args) -> decltype(op(std::forward<decltype(args)>(args)...)) { \
    return op(std::forward<decltype(args)>(args)...);                          \
  }

#define QUERY_OP_CONSTRAINTS(op, device, ...)                                  \
  ::ttnn::graph::query_op_constraints(WRAP_OP(op), device, __VA_ARGS__)

#define QUERY_OP_RUNTIME(op, device, ...)                                      \
  ::ttnn::graph::query_op_runtime(WRAP_OP(op), device, __VA_ARGS__)
// clang-format on

namespace ttnn_op_invoke {

enum class CallType {
  QUERY_OP_CONSTRAINTS,
  QUERY_OP_RUNTIME,
  EXECUTE,
};

} // namespace ttnn_op_invoke

namespace ttnn_op_invoke::operations::utils {

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType);

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout);

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout);

::tt::tt_metal::MathFidelity
toTTNNMathFidelity(::tt::target::MathFidelity mathFidelity);

tt::tt_metal::CoreCoord
toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord);

tt::tt_metal::CoreRange
toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange);

tt::tt_metal::CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSetT &coreRangeSet);

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRefT &tensorRef);

bool inSystemMemory(const ::tt::target::ttnn::TensorRefT &tensorRef);

const ::tt::target::ttnn::MemoryConfigT
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRefT &tensorRef);

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfigT &memcfg);

::ttnn::DeviceComputeKernelConfig createDeviceComputeKernelConfig(
    const ::tt::target::ttnn::DeviceComputeKernelConfigT &config);

} // namespace ttnn_op_invoke::operations::utils

#endif // TTNN_OP_INVOKE_UTILS_H
