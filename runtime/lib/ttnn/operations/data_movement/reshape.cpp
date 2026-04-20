// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"

namespace {

// Tiled reshape can corrupt the cached mapping_tensor's DRAM region when
// the output has non-tile-aligned height or width.  Force the mapping tensor to
// be recreated on every invocation for such shapes.
bool isNonTileAligned(const std::vector<int32_t> &outputShape,
                      const std::array<uint32_t, 2> &tileShape) {
  if (outputShape.empty()) {
    return true;
  }
  int32_t widthDim = outputShape.back();
  int32_t heightDim = outputShape.size() > 1
                          ? outputShape[outputShape.size() - 2]
                          : static_cast<int32_t>(tileShape[0]);
  if (widthDim <= 0 || heightDim <= 0) {
    return true;
  }
  return (static_cast<uint32_t>(widthDim) % tileShape[1] != 0) ||
         (static_cast<uint32_t>(heightDim) % tileShape[0] != 0);
}

::ttnn::TileReshapeMapMode
selectReshapeMapMode(const ::ttnn::Tensor &input,
                     const std::vector<int32_t> &outputShape) {
  const auto &tileShape = input.tensor_spec().tile().get_tile_shape();
  return isNonTileAligned(outputShape, tileShape)
             ? ::ttnn::TileReshapeMapMode::RECREATE
             : ::ttnn::TileReshapeMapMode::CACHE;
}

} // namespace

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  // Workaround for NoC DMA hang.
  // Issue : https://github.com/tenstorrent/tt-metal/issues/40612
  auto mapMode = selectReshapeMapMode(in, shape);
  if (true) {
    mapMode = ::ttnn::TileReshapeMapMode::CACHE;
  }
  ::ttnn::Tensor out = ::ttnn::reshape(in, shape, memoryConfig,
                                       /*padValue=*/std::nullopt, mapMode);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
