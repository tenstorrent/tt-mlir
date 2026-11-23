// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pixel_unshuffle.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/data_movement/pixel_unshuffle/pixel_unshuffle.hpp"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::PixelUnshuffleOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->in());

  // ttnn::pixel_unshuffle wants ROW_MAJOR NCHW. It untilizes internally and
  // accepts sharded input, but the dedicated kernel walks ROW_MAJOR pages, so
  // normalise here rather than relying on the op's internal fallback.
  if (input.memory_config().is_sharded()) {
    ::ttnn::MemoryConfig dramInterleaved{
        ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM};
    input = ::ttnn::to_memory_config(input, dramInterleaved);
  }
  if (input.layout() != ::ttnn::Layout::ROW_MAJOR) {
    input = ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR);
  }

  uint32_t downscaleFactor = op->downscale_factor();
  ::ttnn::PixelUnshuffleChannelOrder channelOrder =
      static_cast<::ttnn::PixelUnshuffleChannelOrder>(op->channel_order());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(
                    op->out()));

  // Derive output_layout from the flatbuffer output spec: tile_shape {1,1} is
  // the ROW_MAJOR sentinel, anything else means TILE.
  const ::tt::target::Dim2d *tileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  const bool outputIsTile =
      tileShape != nullptr && !(tileShape->x() == 1 && tileShape->y() == 1);
  std::optional<::ttnn::Layout> outputLayout =
      outputIsTile ? std::optional<::ttnn::Layout>(::ttnn::Layout::TILE)
                   : std::nullopt;

  // channels_last: NHWC, channel-padded, HEIGHT_SHARDED in L1 (the layout a
  // height-sharded conv2d reads). ttnn rejects a TILE request in this mode.
  const bool channelsLast = op->channels_last();
  std::optional<uint32_t> paddedChannels =
      op->padded_channels() ? std::optional<uint32_t>(op->padded_channels())
                            : std::nullopt;
  if (channelsLast) {
    outputLayout = std::nullopt;
  }

  ::ttnn::Tensor output = ::ttnn::pixel_unshuffle(
      input, downscaleFactor, memoryConfig, outputLayout, channelOrder,
      channelsLast, paddedChannels);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::data_movement
