// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.h"
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "impl/device/device.hpp"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include <optional>

namespace tt::runtime::ttnn::operations::conv {
void run(const ::tt::target::ttnn::Conv2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  DeviceMap &devicePool = context.devicePool;
  ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &weight = tensorPool.at(op->weight()->global_id());
  std::optional<::ttnn::Tensor> bias =
      op->bias() ? std::make_optional(tensorPool.at(op->bias()->global_id()))
                 : std::nullopt;

  auto device = devicePool.at(op->device()->global_id());
  auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
  config.dtype = input.dtype();
  config.weights_dtype = weight.dtype();
  uint32_t N = op->batch_size();
  uint32_t H = op->input_height();
  uint32_t W = op->input_width();
  uint32_t Cin = op->in_channels();
  uint32_t Cout = op->out_channels();

  log_info("INPUT NHWC: {} {} {} {} Cout: {}", N, H, W, Cin, Cout);
  log_info("INPUT DTYPE: {}", input.dtype());
  log_info("WEIGHT DTYPE: {}", input.dtype());

  uint32_t num_cores_x = device->grid_size().x;
  uint32_t num_cores_y = device->grid_size().y;
  // uint32_t num_cores = num_cores_x * num_cores_y;

  if (utils::isOnDevice(input)) {
    auto new_memconfig = input.memory_config();
    if (new_memconfig.memory_layout != TensorMemoryLayout::INTERLEAVED) {
      new_memconfig.memory_layout = TensorMemoryLayout::INTERLEAVED;
      new_memconfig.buffer_type = BufferType::DRAM;
      auto sharded_to_interleaved =
          ::ttnn::operations::data_movement::ShardedToInterleavedOperation();
      input =
          sharded_to_interleaved.invoke(0, input, new_memconfig, input.dtype());
    } else {
      auto new_memconfig = input.memory_config();
      new_memconfig.buffer_type = BufferType::DRAM;
      input = ::ttnn::to_memory_config(input, new_memconfig, std::nullopt);
    }
  }
  // if (N*H*W / num_cores >= 32) {
  //   config.shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
  // }
  if (((((N * H * W) / num_cores_x) >= 8) && ((Cin / num_cores_y) >= 8)) ||
      Cout >= 512) {
    config.shard_layout = TensorMemoryLayout::BLOCK_SHARDED;
  }
  // else if ((Cin)){
  //   config.shard_layout = TensorMemoryLayout::WIDTH_SHARDED;
  // }

  ::ttnn::Tensor out =
      std::get<0>(::ttnn::operations::conv::conv2d::conv2d<::ttnn::Device>(
          input, weight, device, op->in_channels(), op->out_channels(),
          op->batch_size(), op->input_height(), op->input_width(),
          {op->kernel_height(), op->kernel_width()},
          {op->stride_height(), op->stride_width()},
          {op->padding_height(), op->padding_width()},
          {op->dilation_height(), op->dilation_width()}, op->groups(), bias,
          config));

  log_info("OUT DTYPE: {}", out.dtype());
  auto new_memconfig = out.memory_config();
  new_memconfig.memory_layout = TensorMemoryLayout::INTERLEAVED;
  // new_memconfig.buffer_type = BufferType::DRAM;
  auto sharded_to_interleaved =
      ::ttnn::operations::data_movement::ShardedToInterleavedOperation();
  out = sharded_to_interleaved.invoke(0, out, new_memconfig, out.dtype());
  new_memconfig.buffer_type = BufferType::DRAM;
  out = ::ttnn::to_memory_config(out, new_memconfig, std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::conv
