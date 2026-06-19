// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/pool2d.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/Pool/Pool2dOp.h"
#include "ttmlir/Target/TTNN/operations/pool_generated.h"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include <ttnn/operations/functions.hpp>
#include <ttnn/operations/pool/generic/generic_pools.hpp>
#include <variant>

namespace tt::runtime::ttnn::operations::pool {

void runAvgPool2dOp(const ::tt::target::ttnn::Pool2dOp *op,
                    ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::Pool2dOpT pool2dOpNative;
  op->UnPackTo(&pool2dOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::AvgPool2dOpResult result = ttnn_op_invoke::callAvgPool2d(
      ttnn_op_invoke::CallType::EXECUTE, pool2dOpNative, &input, &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callAvgPool2d execution");

  ::ttnn::Tensor out = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void runMaxPool2dOp(const ::tt::target::ttnn::Pool2dOp *op,
                    ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::Pool2dOpT pool2dOpNative;
  op->UnPackTo(&pool2dOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::MaxPool2dOpResult result = ttnn_op_invoke::callMaxPool2d(
      ttnn_op_invoke::CallType::EXECUTE, pool2dOpNative, &input, &targetDevice);

  LOG_ASSERT(std::holds_alternative<std::vector<::ttnn::Tensor>>(result),
             "Expected Tensor from callMaxPool2d execution");

  ::ttnn::Tensor output = std::get<std::vector<::ttnn::Tensor>>(result)[0];
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::Pool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::Pool2dOpType::AvgPool2d: {
    runAvgPool2dOp(op, tensorPool, context);
    break;
  }
  case ::tt::target::ttnn::Pool2dOpType::MaxPool2d: {
    runMaxPool2dOp(op, tensorPool, context);
    break;
  }
  }
}

void run(const ::tt::target::ttnn::MaxPool2dWithIndicesOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::MaxPool2dWithIndicesOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::MaxPool2dWithIndicesOpResult result =
      ttnn_op_invoke::callMaxPool2dWithIndices(
          ttnn_op_invoke::CallType::EXECUTE, opNative, &input, &targetDevice);

  LOG_ASSERT(std::holds_alternative<std::vector<::ttnn::Tensor>>(result),
             "Expected vector<Tensor> from callMaxPool2dWithIndices execution");

  const auto &outputs = std::get<std::vector<::ttnn::Tensor>>(result);
  tensorPool.insertTTNNTensorAndValidate(op->result(), outputs[0]);
  tensorPool.insertTTNNTensorAndValidate(op->result_indices(), outputs[1]);
}

void run(const ::tt::target::ttnn::GlobalAvgPool2dOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::DataType dtype = input.dtype();
  if (op->dtype()) {
    dtype = ttnn_op_invoke::operations::utils::toTTNNDataType(*op->dtype());
  }

  auto inputShape = input.logical_shape();
  LOG_ASSERT(inputShape.rank() == 4,
             "GlobalAvgPool2d expects a rank 4 input tensor");
  uint32_t batchSize = inputShape[0];
  uint32_t inputHeight = inputShape[1];
  uint32_t inputWidth = inputShape[2];
  uint32_t inputChannels = inputShape[3];

  ::ttnn::Layout outputLayout =
      ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());

  ::ttnn::Tensor out = ::ttnn::avg_pool2d(
      input, batchSize, inputHeight, inputWidth, inputChannels,
      /*kernel_size=*/{inputHeight, inputWidth},
      /*stride=*/{1, 1}, /*padding=*/std::array<uint32_t, 2>{0, 0},
      /*ceil_mode=*/false, /*count_include_pad=*/true,
      /*divisor_override=*/std::nullopt, outputMemoryConfig,
      /*dram_slice_config=*/std::nullopt,
      /*applied_shard_scheme=*/std::nullopt,
      /*compute_kernel_config=*/std::nullopt,
      /*deallocate_input=*/false,
      /*reallocate_halo_output=*/true, dtype, outputLayout,
      /*config_tensor_in_dram=*/false);

  out = ::ttnn::reshape(out, ::ttnn::Shape({batchSize, 1, 1, inputChannels}),
                        outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::pool
