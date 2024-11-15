// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <stdexcept>
#include <tuple>

#include <llvm/Support/Casting.h>
#include <mlir/IR/AttrTypeSubElements.h>

#include "TTNNOpModel.hpp"
#include "TTNNOpModelLib_Impl.hpp"
#include "impl/buffers/buffer.hpp"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttnn/tensor/types.hpp"

namespace mlir::tt::op_model::ttnn {

// alias to a common tt_metal types
using DataType = ::tt::tt_metal::DataType;
using Layout = ::tt::tt_metal::Layout;
using CoreRange = ::tt::tt_metal::CoreRange;
using CoreRangeSet = ::tt::tt_metal::CoreRangeSet;
using CoreCoord = ::tt::tt_metal::CoreCoord;
using ShardSpec = ::tt::tt_metal::ShardSpec;
using ShardOrientation = ::tt::tt_metal::ShardOrientation;
using TensorMemoryLayout = ::tt::tt_metal::TensorMemoryLayout;
using MemoryConfig = ::tt::tt_metal::MemoryConfig;

namespace detail {

DataType getDataType(const mlir::MemRefType &memref) {

  auto dataType = elementTypeToDataType(memref.getElementType());

  switch (dataType) {
  case tt::DataType::Float32:
    return DataType::FLOAT32;
  case tt::DataType::BFloat16:
    return DataType::BFLOAT16;
  case tt::DataType::BFP_BFloat8:
    return DataType::BFLOAT8_B;
  case tt::DataType::BFP_BFloat4:
    return DataType::BFLOAT4_B;
  case tt::DataType::UInt32:
    return DataType::UINT32;
  case tt::DataType::UInt16:
    return DataType::UINT16;
  case tt::DataType::UInt8:
    return DataType::UINT8;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

::ttnn::SimpleShape getTensorShape(const mlir::MemRefType &memref) {
  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape(
      memref.getShape().begin(), memref.getShape().end());
  return ::ttnn::SimpleShape(small_vector_shape);
}

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::LayoutAttr &layout) {
  const auto layoutShardTile = layout.getShardShape(false);

  if (layoutShardTile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shardShape;
  shardShape[0] = layoutShardTile[0];
  shardShape[1] = layoutShardTile[1];
  return shardShape;
}

Layout getTensorLayout(const mlir::tt::LayoutAttr &layout) {
  return layout.isTiled() ? Layout::TILE : Layout::ROW_MAJOR;
}

CoreRangeSet getCoreRangeSet(const mlir::tt::LayoutAttr &layout) {
  // TODO(mbezulj): handle more complex grid shapes
  // assuming grid shape is one rect starting at (0,0)

  const auto layoutGrid = layout.getGrid();

  const auto layoutGridShape = layoutGrid.getShape();
  if (layoutGridShape.size() != 2) {
    llvm::errs() << "ERROR: layout_grid.getShape().size() == 2\n";
    return {};
  }

  return CoreRangeSet(CoreRange(CoreCoord(0, layoutGridShape[0]),
                                CoreCoord(0, layoutGridShape[1])));
}

std::optional<ShardSpec>
layout_get_shard_spec(const mlir::tt::LayoutAttr &layout) {
  // tt_ShardOrientation is not part of LayoutAttr;
  // defaulting to ROW_MAJOR. TODO: figure out if we need to expose this
  return isShardedMemoryLayout(layout.getMemLayout())
             ? std::make_optional(ShardSpec(getCoreRangeSet(layout),
                                            getShardShape(layout),
                                            ShardOrientation::ROW_MAJOR, false))
             : std::nullopt;
}

::tt::tt_metal::BufferType getBufferType(const mlir::MemRefType &memref) {
  auto memorySpace =
      mlir::cast<tt::MemorySpaceAttr>(memref.getMemorySpace()).getValue();

  switch (memorySpace) {
  case tt::MemorySpace::DeviceDRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case tt::MemorySpace::DeviceL1:
    return ::tt::tt_metal::BufferType::L1;
  default: // TODO(mbezulj): handle other memory spaces
    throw std::runtime_error("Unsupported memory space");
  }
}

TensorMemoryLayout getTensorMemoryLayout(const mlir::tt::LayoutAttr &layout) {
  auto tensorMemoryLayout = layout.getMemLayout();

  switch (tensorMemoryLayout) {
  case tt::TensorMemoryLayout::Interleaved:
    return TensorMemoryLayout::INTERLEAVED;
  case tt::TensorMemoryLayout::SingleBank:
    return TensorMemoryLayout::SINGLE_BANK;
  case tt::TensorMemoryLayout::HeightSharded:
    return TensorMemoryLayout::HEIGHT_SHARDED;
  case tt::TensorMemoryLayout::WidthSharded:
    return TensorMemoryLayout::WIDTH_SHARDED;
  case tt::TensorMemoryLayout::BlockSharded:
    return TensorMemoryLayout::BLOCK_SHARDED;
  default:
    throw std::runtime_error("Unsupported tensor memory layout");
  }
}

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::LayoutAttr &layout) {

  auto tensorMemoryLayout = getTensorMemoryLayout(layout);
  auto bufferType = getBufferType(layout.getMemref());

  auto shardSpec = layout_get_shard_spec(layout);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

} // namespace detail

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

bool ReluOpInterface::isLegal(const mlir::tt::LayoutAttr &inputLayout,
                              const mlir::tt::LayoutAttr &outputLayout) {
  return true; // to solve when we have metal implementation
}

std::tuple<size_t, size_t, size_t>
ReluOpInterface::getOpL1Usage(const mlir::tt::LayoutAttr &inputLayout,
                              const mlir::tt::LayoutAttr &outputLayout) {
  return std::make_tuple(0, 0, 0); // to solve when we have metal implementation
}

} // namespace mlir::tt::op_model::ttnn
