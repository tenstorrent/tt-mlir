// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pixel_shuffle.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/data_movement/pixel_shuffle/pixel_shuffle.hpp"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::PixelShuffleOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());
  uint32_t upscaleFactor = op->upscale_factor();
  ::ttnn::PixelUnshuffleChannelOrder channelOrder =
      static_cast<::ttnn::PixelUnshuffleChannelOrder>(op->channel_order());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  if (op->channels_last()) {
    // The NHWC kernel reads the row-major height-sharded activation and
    // writes a row-major height-sharded output whose shard is
    // [pixels_per_core, padded_channels]; the compiler put that memory
    // config on the op and untilizes the input.
    LOG_ASSERT(memoryConfig.has_value(),
               "PixelShuffleOp(channels_last) requires a memory config");
    if (input.layout() != ::ttnn::Layout::ROW_MAJOR) {
      input = ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR);
    }
    std::optional<uint32_t> paddedChannels =
        op->padded_channels() ? std::optional<uint32_t>(op->padded_channels())
                              : std::nullopt;
    ::ttnn::Tensor output = ::ttnn::pixel_shuffle(
        input, upscaleFactor, memoryConfig, /*output_layout=*/std::nullopt,
        channelOrder, /*channels_last=*/true, paddedChannels);
    tensorPool.insertTTNNTensorAndValidate(op->out(), output);
    return;
  }
  const ::tt::target::Dim2d *tileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  const bool outputIsTile =
      tileShape != nullptr && !(tileShape->x() == 1 && tileShape->y() == 1);
  std::optional<::ttnn::Layout> outputLayout =
      outputIsTile ? std::optional<::ttnn::Layout>(::ttnn::Layout::TILE)
                   : std::nullopt;
  ::ttnn::Tensor output = ::ttnn::pixel_shuffle(
      input, upscaleFactor, memoryConfig, outputLayout, channelOrder);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::data_movement
