// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/maxpool2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/types.hpp"
#include <optional>

namespace tt::runtime::ttnn::operations::pool {

// TODO(bug #855): Ideally we should have an op that preshards for maxpool2d
// instead of adding a method in runtime
template <typename DeviceType>
static ::ttnn::Tensor
preshardForMaxPool2d(const ::tt::target::ttnn::MaxPool2dOp *op,
                     DeviceType &device, const ::ttnn::Tensor &input) {
  const ::ttnn::SimpleShape inputShape{
      ::tt::runtime::ttnn::utils::toShapeFromFBShape(
          *op->in()->desc()->shape())};
  uint32_t output_height =
      1 + (op->input_height() + 2 * op->padding_height() -
           op->dilation_height() * (op->kernel_height() - 1) - 1) /
              op->stride_height();
  uint32_t output_width =
      1 + (op->input_width() + 2 * op->padding_width() -
           op->dilation_width() * (op->kernel_width() - 1) - 1) /
              op->stride_width();

  constexpr bool en_ch_padding = false;

  auto parallel_config = ::ttnn::operations::conv::determine_parallel_config(
      ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, op->batch_size(),
      op->channels(), output_height, output_width, op->channels(),
      device.compute_with_storage_grid_size(), ShardOrientation::ROW_MAJOR,
      en_ch_padding);
  auto sharded_memory_config = ::ttnn::operations::conv::
      create_sharded_memory_config_from_parallel_config(inputShape,
                                                        parallel_config, 1);
  return ::ttnn::to_memory_config(input, sharded_memory_config, std::nullopt);
}

void run(const ::tt::target::ttnn::MaxPool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::operations::pool::Pool2DOp<
      ::ttnn::operations::pool::Pool2DType::MAX_POOL2D>
      operation = ::ttnn::operations::pool::Pool2DOp<
          ::ttnn::operations::pool::Pool2DType::MAX_POOL2D>();

  ::ttnn::Tensor input = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(input.is_allocated());
  if (workaround::Env::get().maxpool2dPreshard) {
    DeviceVariant targetDevice =
        context.getTargetDevice(op->device()->global_id());
    input = std::visit(
        [&](auto &&targetDevice) -> ::ttnn::Tensor {
          return preshardForMaxPool2d(op, targetDevice.get(), input);
        },
        targetDevice);
  }
  ::ttnn::MemoryConfig outMemConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  ::ttnn::Tensor out = operation.invoke(
      0, input, op->batch_size(), op->input_height(), op->input_width(),
      op->channels(), {op->kernel_height(), op->kernel_width()},
      {op->stride_height(), op->stride_width()},
      {op->padding_height(), op->padding_width()},
      {op->dilation_height(), op->dilation_width()}, outMemConfig,
      std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::pool
