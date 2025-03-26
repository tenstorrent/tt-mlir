// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <stdexcept>
#ifdef TTMLIR_ENABLE_OPMODEL
#include "Conversion.hpp"

#include "ttmlir/Dialect/TT/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::op_model::ttnn {

namespace conversion {

::tt::tt_metal::DataType getDataType(const DataType dataType) {
  switch (dataType) {
  case tt::DataType::Float32:
    return ::tt::tt_metal::DataType::FLOAT32;
  case tt::DataType::BFloat16:
    return ::tt::tt_metal::DataType::BFLOAT16;
  case tt::DataType::BFP_BFloat8:
    return ::tt::tt_metal::DataType::BFLOAT8_B;
  case tt::DataType::BFP_BFloat4:
    return ::tt::tt_metal::DataType::BFLOAT4_B;
  case tt::DataType::UInt32:
    return ::tt::tt_metal::DataType::UINT32;
  case tt::DataType::UInt16:
    return ::tt::tt_metal::DataType::UINT16;
  case tt::DataType::UInt8:
    return ::tt::tt_metal::DataType::UINT8;
  case tt::DataType::Int32:
    return ::tt::tt_metal::DataType::INT32;
  default:
    throw std::runtime_error("Invalid element type");
  }
}

::ttnn::Shape getShape(const ::llvm::ArrayRef<int64_t> shape) {
  ::tt::stl::SmallVector<uint32_t> small_vector_shape;
  for (const auto &dim : shape) {
    small_vector_shape.push_back(static_cast<uint32_t>(dim));
  }

  return ::ttnn::Shape(small_vector_shape);
}

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  const auto layoutShardTile = layout.getScalarShardShape();

  if (layoutShardTile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shardShape;
  shardShape[0] = layoutShardTile[0];
  shardShape[1] = layoutShardTile[1];
  return shardShape;
}

::tt::tt_metal::Layout
getPageLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return layout.isTiled() ? ::tt::tt_metal::Layout::TILE
                          : ::tt::tt_metal::Layout::ROW_MAJOR;
}

::tt::tt_metal::Layout getPageLayout(mlir::tt::ttnn::Layout layout) {
  switch (layout) {
  case ::mlir::tt::ttnn::Layout::RowMajor:
    return ::tt::tt_metal::Layout::ROW_MAJOR;
  case ::mlir::tt::ttnn::Layout::Tile:
    return ::tt::tt_metal::Layout::TILE;
  case ::mlir::tt::ttnn::Layout::Invalid:
    return ::tt::tt_metal::Layout::INVALID;
  }
}

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  std::set<::tt::tt_metal::CoreRange> coreRangeSet;
  assert(layout.getGrid().getMapping().isEmpty() == false);
  for (const auto &[loc, size] : utils::toCoreRangeSet(
           layout.getGrid().getShape(), layout.getGrid().getMapping())) {
    coreRangeSet.insert(::tt::tt_metal::CoreRange(
        CoreCoord(loc[0], loc[1]),
        CoreCoord(loc[0] + size[0] - 1, loc[1] + size[1] - 1)));
  }
  return ::tt::tt_metal::CoreRangeSet(coreRangeSet);
}

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  // tt_ShardOrientation is not part of ttnn::TTNNLayoutAttr;
  // defaulting to ROW_MAJOR. TODO(jserbedzija): with issue #620
  return isShardedMemoryLayout(layout.getMemLayout().getValue())
             ? std::make_optional(::tt::tt_metal::ShardSpec(
                   getCoreRangeSet(layout), getShardShape(layout),
                   ::tt::tt_metal::ShardOrientation::ROW_MAJOR))
             : std::nullopt;
}

::tt::tt_metal::BufferType
getBufferType(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto bufferType = layout.getBufferType();

  switch (bufferType) {
  case mlir::tt::ttnn::BufferType::DRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case mlir::tt::ttnn::BufferType::L1:
    return ::tt::tt_metal::BufferType::L1;
  case mlir::tt::ttnn::BufferType::SystemMemory:
    return ::tt::tt_metal::BufferType::SYSTEM_MEMORY;
  case mlir::tt::ttnn::BufferType::L1Small:
    return ::tt::tt_metal::BufferType::L1_SMALL;
  case mlir::tt::ttnn::BufferType::Trace:
    return ::tt::tt_metal::BufferType::TRACE;
  }
}

::tt::tt_metal::TensorMemoryLayout getTensorMemoryLayout(
    const mlir::tt::ttnn::TensorMemoryLayoutAttr memLayoutAttr) {
  auto tensorMemoryLayout = memLayoutAttr.getValue();

  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::Interleaved:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  case mlir::tt::ttnn::TensorMemoryLayout::SingleBank:
    return ::tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
    return ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
    return ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
    return ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
  }
}

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto tensorMemoryLayout = getTensorMemoryLayout(layout.getMemLayout());
  auto bufferType = getBufferType(layout);

  auto shardSpec = getShardSpec(layout);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

::tt::tt_metal::TensorLayout
getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return ::tt::tt_metal::TensorLayout(getDataType(layout.getDataType()),
                                      getPageLayout(layout),
                                      getMemoryConfig(layout));
}

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return ::ttnn::TensorSpec(getShape(shape), getTensorLayout(layout));
}

::ttnn::SmallVector<int>
convertLLVMSmallVecToTTNNSmallVec(const ::llvm::ArrayRef<int64_t> vec) {
  return ::ttnn::SmallVector<int>(vec.begin(), vec.end());
}

std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig> getConv2dConfig(
    const std::optional<mlir::tt::ttnn::Conv2dConfigAttr> &conv2dConfig) {
  if (!conv2dConfig) {
    return std::nullopt;
  }

  // TODO(#2130): config.core_grid is hardcoded to nullopt until we add
  // CoreRangeSet as an IR attribute.
  assert(!conv2dConfig->getCoreGrid() && "CoreGrid is not supported yet");

  ::ttnn::operations::conv::conv2d::Conv2dConfig config;
  config.dtype = getDataType(conv2dConfig->getDtype());
  config.weights_dtype = getDataType(conv2dConfig->getWeightsDtype());
  config.activation = conv2dConfig->getActivation().str();
  config.input_channels_alignment =
      conv2dConfig->getInputChannelsAlignment().getInt();
  config.deallocate_activation =
      conv2dConfig->getDeallocateActivation().getValue();
  config.reallocate_halo_output =
      conv2dConfig->getReallocateHaloOutput().getValue();
  config.act_block_h_override = conv2dConfig->getActBlockHOverride().getInt();
  config.act_block_w_div = conv2dConfig->getActBlockWDiv().getInt();
  config.reshard_if_not_optimal =
      conv2dConfig->getReshardIfNotOptimal().getValue();
  config.override_sharding_config =
      conv2dConfig->getOverrideShardingConfig().getValue();
  config.shard_layout = conv2dConfig->getShardLayout()
                            ? std::make_optional(getTensorMemoryLayout(
                                  conv2dConfig->getShardLayout()))
                            : std::nullopt;
  config.core_grid = std::nullopt;
  config.transpose_shards = conv2dConfig->getTransposeShards().getValue();
  config.output_layout = getPageLayout(conv2dConfig->getOutputLayout());
  config.enable_act_double_buffer =
      conv2dConfig->getEnableActDoubleBuffer().getValue();
  config.enable_weights_double_buffer =
      conv2dConfig->getEnableWeightsDoubleBuffer().getValue();
  config.enable_split_reader = conv2dConfig->getEnableSplitReader().getValue();
  config.enable_subblock_padding =
      conv2dConfig->getEnableSubblockPadding().getValue();

  return config;
}

} // namespace conversion
} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_ENABLE_OPMODEL
