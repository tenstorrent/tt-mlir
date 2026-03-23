// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/normalization/layernorm_pre_allgather.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/dit_layernorm_pre_all_gather.hpp"

namespace tt::runtime::ttnn::operations::layernorm_pre_allgather {
void run(const ::tt::target::ttnn::LayerNormPreAllGatherOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &recipTensor =
      tensorPool.getTTNNTensorAndValidate(op->recip_tensor());

  // Handle optional dtype, default to BFLOAT16.
  ::ttnn::DataType dtype = ::ttnn::DataType::BFLOAT16;
  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype().value());
  }

  // Handle optional compute kernel config.
  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig = std::nullopt;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  // Handle optional memory config.
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  ::ttnn::Tensor output = ::ttnn::experimental::dit_layernorm_pre_allgather(
      input, recipTensor, dtype, computeConfig, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::layernorm_pre_allgather
