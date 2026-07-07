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

  // ttnn::pixel_unshuffle requires ROW_MAJOR NCHW input.
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
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));

  // Derive output_layout from the flatbuffer output tensor spec.
  // tile_shape {1,1} is the ROW_MAJOR sentinel; any other valid tile means TILE.
  const ::tt::target::Dim2d *tileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  const bool outputIsTile =
      tileShape != nullptr && !(tileShape->x() == 1 && tileShape->y() == 1);
  std::optional<::ttnn::Layout> outputLayout =
      outputIsTile ? std::optional<::ttnn::Layout>(::ttnn::Layout::TILE)
                   : std::nullopt;

  ::ttnn::Tensor output =
      ::ttnn::pixel_unshuffle(input, downscaleFactor, memoryConfig, outputLayout,
                              channelOrder);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::data_movement
