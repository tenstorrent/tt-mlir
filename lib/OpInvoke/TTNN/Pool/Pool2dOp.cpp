// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/Pool/Pool2dOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttnn/operations/pool/generic/generic_pools.hpp"

#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <optional>

namespace ttnn_op_invoke {

Pool2dResolvedParams
resolvePool2dParams(const ::tt::target::ttnn::Pool2dOpT &op) {
  Pool2dResolvedParams params;

  TT_INVOKE_ASSERT(op.kernel_size.size() == 2,
                   "kernel_size must have 2 elements");
  TT_INVOKE_ASSERT(op.stride.size() == 2, "stride must have 2 elements");

  std::copy_n(op.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(op.stride.begin(), 2, params.stride.begin());

  TT_INVOKE_ASSERT(op.padding.size() == 2 || op.padding.size() == 4,
                   "padding must have 2 or 4 elements");
  if (op.padding.size() == 2) {
    params.padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op.padding[0]),
                                static_cast<uint32_t>(op.padding[1])};
  } else {
    params.padding =
        std::array<uint32_t, 4>{static_cast<uint32_t>(op.padding[0]),  // top
                                static_cast<uint32_t>(op.padding[2]),  // bottom
                                static_cast<uint32_t>(op.padding[1]),  // left
                                static_cast<uint32_t>(op.padding[3])}; // right
  }

  if (op.applied_shard_scheme) {
    params.appliedShardScheme =
        operations::utils::toTTNNTensorMemoryLayout(*op.applied_shard_scheme);
  }

  params.dtype = ::ttnn::DataType::BFLOAT16;
  params.output_layout = ::ttnn::Layout::ROW_MAJOR;

  if (op.type == ::tt::target::ttnn::Pool2dOpType::AvgPool2d) {
    params.countIncludePad = std::make_optional(false);
    const tt::target::ttnn::AvgPool2dExtraParamsT *avgParams =
        op.extra_params.AsAvgPool2dExtraParams();
    params.countIncludePad = avgParams->count_include_pad;
  } else if (op.type == ::tt::target::ttnn::Pool2dOpType::MaxPool2d) {
    TT_INVOKE_ASSERT(op.dilation.size() == 2, "dilation must have 2 elements");
    std::copy(op.dilation.begin(), op.dilation.end(),
              params.dilation.emplace().begin());
  } else {
    llvm_unreachable("unhandled Pool2dOpType");
  }

  if (op.out) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.out));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.out) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createAvgPool2dTuple(Tag tag, const ::tt::target::ttnn::Pool2dOpT &op,
                          TensorArg input, const Pool2dResolvedParams &params) {
  TT_INVOKE_ASSERT(params.countIncludePad.has_value(),
                   "countIncludePad must be resolved for AvgPool2dOp");
  return std::make_tuple(
      resolveTensorArg(input, tag), op.batch_size, op.input_height,
      op.input_width, op.channels, params.kernelSize, params.stride,
      params.padding, op.ceil_mode, *params.countIncludePad,
      /*divisor_override=*/std::nullopt, params.outputMemoryConfig,
      /*dram_slice_config=*/std::nullopt, params.appliedShardScheme,
      /*compute_kernel_config=*/std::nullopt,
      /*deallocate_input=*/false, op.reallocate_halo_output, params.dtype,
      params.output_layout, op.config_tensors_in_dram);
}

AvgPool2dOpResult callAvgPool2d(CallType callType,
                                const ::tt::target::ttnn::Pool2dOpT &op,
                                TensorArg input, ::ttnn::MeshDevice *device) {
  TT_INVOKE_ASSERT(op.type == ::tt::target::ttnn::Pool2dOpType::AvgPool2d,
                   "Expected AvgPool2dOpType");

  Pool2dResolvedParams params = resolvePool2dParams(op);

  auto makeTuple = [&](auto tag) {
    return createAvgPool2dTuple(tag, op, input, params);
  };

  return callOp<AvgPool2dOpResult>(WRAP_OP(::ttnn::avg_pool2d), callType,
                                   makeTuple, device);
}

template <typename Tag>
auto createMaxPool2dTuple(Tag tag, const ::tt::target::ttnn::Pool2dOpT &op,
                          TensorArg input, const Pool2dResolvedParams &params) {
  TT_INVOKE_ASSERT(params.dilation.has_value(),
                   "dilation must be resolved for MaxPool2dOp");
  return std::make_tuple(
      resolveTensorArg(input, tag), op.batch_size, op.input_height,
      op.input_width, op.channels, params.kernelSize, params.stride,
      params.padding, *params.dilation, op.ceil_mode, params.outputMemoryConfig,
      /*dram_slice_config=*/std::nullopt, params.appliedShardScheme,
      /*deallocate_input=*/false, op.reallocate_halo_output,
      /*return_indices=*/false, params.dtype, params.output_layout,
      op.config_tensors_in_dram);
}

MaxPool2dOpResult callMaxPool2d(CallType callType,
                                const ::tt::target::ttnn::Pool2dOpT &op,
                                TensorArg input, ::ttnn::MeshDevice *device) {
  TT_INVOKE_ASSERT(op.type == ::tt::target::ttnn::Pool2dOpType::MaxPool2d,
                   "Expected MaxPool2dOpType");

  Pool2dResolvedParams params = resolvePool2dParams(op);

  auto makeTuple = [&](auto tag) {
    return createMaxPool2dTuple(tag, op, input, params);
  };

  return callOp<MaxPool2dOpResult>(WRAP_OP(::ttnn::max_pool2d), callType,
                                   makeTuple, device);
}

Pool2dResolvedParams resolveMaxPool2dWithIndicesParams(
    const ::tt::target::ttnn::MaxPool2dWithIndicesOpT &op) {
  Pool2dResolvedParams params;

  TT_INVOKE_ASSERT(op.kernel_size.size() == 2,
                   "kernel_size must have 2 elements");
  TT_INVOKE_ASSERT(op.stride.size() == 2, "stride must have 2 elements");
  TT_INVOKE_ASSERT(op.dilation.size() == 2, "dilation must have 2 elements");

  std::copy_n(op.kernel_size.begin(), 2, params.kernelSize.begin());
  std::copy_n(op.stride.begin(), 2, params.stride.begin());
  std::copy(op.dilation.begin(), op.dilation.end(),
            params.dilation.emplace().begin());

  TT_INVOKE_ASSERT(op.padding.size() == 2 || op.padding.size() == 4,
                   "padding must have 2 or 4 elements");
  if (op.padding.size() == 2) {
    params.padding =
        std::array<uint32_t, 2>{static_cast<uint32_t>(op.padding[0]),
                                static_cast<uint32_t>(op.padding[1])};
  } else {
    params.padding =
        std::array<uint32_t, 4>{static_cast<uint32_t>(op.padding[0]),  // top
                                static_cast<uint32_t>(op.padding[2]),  // bottom
                                static_cast<uint32_t>(op.padding[1]),  // left
                                static_cast<uint32_t>(op.padding[3])}; // right
  }

  if (op.applied_shard_scheme) {
    params.appliedShardScheme =
        operations::utils::toTTNNTensorMemoryLayout(*op.applied_shard_scheme);
  }

  params.dtype = ::ttnn::DataType::BFLOAT16;
  params.output_layout = ::ttnn::Layout::ROW_MAJOR;

  if (op.result) {
    params.outputMemoryConfig = operations::utils::createMemoryConfigIfNeeded(
        operations::utils::getTensorRefMemoryConfig(*op.result));
    TT_INVOKE_ASSERT(operations::utils::inSystemMemory(*op.result) ||
                         params.outputMemoryConfig.has_value(),
                     "Memory config must exist for device tensors");
  }

  return params;
}

template <typename Tag>
auto createMaxPool2dWithIndicesTuple(
    Tag tag, const ::tt::target::ttnn::MaxPool2dWithIndicesOpT &op,
    TensorArg input, const Pool2dResolvedParams &params) {
  TT_INVOKE_ASSERT(params.dilation.has_value(),
                   "dilation must be resolved for MaxPool2dWithIndicesOp");
  return std::make_tuple(
      resolveTensorArg(input, tag), op.batch_size, op.input_height,
      op.input_width, op.channels, params.kernelSize, params.stride,
      params.padding, *params.dilation, op.ceil_mode, params.outputMemoryConfig,
      /*dram_slice_config=*/std::nullopt, params.appliedShardScheme,
      /*deallocate_input=*/false, op.reallocate_halo_output,
      /*return_indices=*/true, params.dtype, params.output_layout,
      op.config_tensors_in_dram);
}

MaxPool2dWithIndicesOpResult
callMaxPool2dWithIndices(CallType callType,
                         const ::tt::target::ttnn::MaxPool2dWithIndicesOpT &op,
                         TensorArg input, ::ttnn::MeshDevice *device) {
  Pool2dResolvedParams params = resolveMaxPool2dWithIndicesParams(op);

  auto makeTuple = [&](auto tag) {
    return createMaxPool2dWithIndicesTuple(tag, op, input, params);
  };

  return callOp<MaxPool2dWithIndicesOpResult>(WRAP_OP(::ttnn::max_pool2d),
                                              callType, makeTuple, device);
}

} // namespace ttnn_op_invoke
