// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.h"
#include "impl/buffers/buffer_constants.hpp"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/core/core.hpp"

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  // TODO (jnie): Update this once we support multi device tensors
  // Investigate how to handle multi device in conv2d
  ::ttnn::Device &device =
      context.getDeviceFromView(op->device()->global_id(), 0);
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;
  auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
  config.dtype = utils::getDataType(op->input());
  config.weights_dtype = utils::getDataType(op->weight());

  // This metric is used to determine whether to use BLOCK_SHARDED or
  // to just use the default HEIGHT_SHARDED. Truly modelling what sharding
  // config to use seems to be an open question.
  // Metal issue: https://github.com/tenstorrent/tt-metal/issues/13107
  // MLIR issue: https://github.com/tenstorrent/tt-mlir/issues/830
  if (op->in_channels() / device.grid_size().y >= 32) {
    config.shard_layout = TensorMemoryLayout::BLOCK_SHARDED;
  }

  ::ttnn::Tensor out =
      std::get<0>(::ttnn::operations::conv::conv2d::conv2d<::ttnn::Device>(
          input, weight, &device, op->in_channels(), op->out_channels(),
          op->batch_size(), op->input_height(), op->input_width(),
          {op->kernel_height(), op->kernel_width()},
          {op->stride_height(), op->stride_width()},
          {op->padding_height(), op->padding_width()},
          {op->dilation_height(), op->dilation_width()}, op->groups(), bias,
          config));

  // Workaround. The compiler models all ops as outputting to
  // DRAM - interleaved. The compiler is not yet setup to model
  // memory configs. TTNN::conv2d outputs to L1, in some sharded
  // config. In order to ensure the compiler models the memory config
  // "correctly", I put to_memory_config here. In addition to the
  // compiler being "correct", subsequent eltwise ops require that both
  // inputs be on DRAM - interleaved.
  //
  // Issue: https://github.com/tenstorrent/tt-mlir/issues/826
  auto new_memconfig = out.memory_config();
  new_memconfig.memory_layout = TensorMemoryLayout::INTERLEAVED;
  new_memconfig.buffer_type = BufferType::DRAM;
  out = ::ttnn::to_memory_config(out, new_memconfig, std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
