// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "TTNNOpModel.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "Conversion.hpp"
#include "ttmlir/Dialect/TT/Utils/CoreRangeSet.h"

namespace mlir::tt::op_model::ttnn {

namespace conversion {

::tt::tt_metal::DataType
getDataType(const mlir::tt::ttnn::TTNNLayoutAttr layout) {
  auto dataType = layout.getDataType();

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
  default:
    throw std::runtime_error("Invalid element type");
  }
}

::ttnn::SimpleShape getSimpleShape(const ::llvm::ArrayRef<int64_t> &shape) {
  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape;
  for (const auto &dim : shape) {
    small_vector_shape.push_back(static_cast<uint32_t>(dim));
  }

  return ::ttnn::SimpleShape(small_vector_shape);
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

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                const mlir::tt::GridAttr &workerGrid) {
  std::set<::tt::tt_metal::CoreRange> coreRangeSet;
  for (const auto &[loc, size] : toCoreRangeSet(layout.getGrid(), workerGrid)) {
    coreRangeSet.insert(::tt::tt_metal::CoreRange(
        CoreCoord(loc[0], loc[1]),
        CoreCoord(loc[0] + size[0] - 1, loc[1] + size[1] - 1)));
  }
  return ::tt::tt_metal::CoreRangeSet(coreRangeSet);
}

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
             const mlir::tt::GridAttr &workerGrid) {
  // tt_ShardOrientation is not part of ttnn::TTNNLayoutAttr;
  // defaulting to ROW_MAJOR. TODO: figure out if we need to expose this
  return isShardedMemoryLayout(layout.getMemLayout().getValue())
             ? std::make_optional(ShardSpec(getCoreRangeSet(layout, workerGrid),
                                            getShardShape(layout),
                                            ShardOrientation::ROW_MAJOR, false))
             : std::nullopt;
}

::tt::tt_metal::BufferType getBufferType(const mlir::MemRefType &memref) {
  auto memorySpace =
      mlir::cast<mlir::tt::ttnn::BufferTypeAttr>(memref.getMemorySpace())
          .getValue();

  switch (memorySpace) {
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

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto tensorMemoryLayout = layout.getMemLayout().getValue();

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
getMemoryConfig(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                const mlir::tt::GridAttr &workerGrid) {

  auto tensorMemoryLayout = getTensorMemoryLayout(layout);
  auto bufferType = getBufferType(layout.getMemref());

  auto shardSpec = getShardSpec(layout, workerGrid);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

::tt::tt_metal::TensorLayout
getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                const mlir::tt::GridAttr &workerGrid) {
  return ::tt::tt_metal::TensorLayout(getDataType(layout),
                                      getPageLayout(layout),
                                      getMemoryConfig(layout, workerGrid));
}

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const mlir::tt::ttnn::TTNNLayoutAttr &layout,
                                 const mlir::tt::GridAttr &workerGrid) {
  return ::ttnn::TensorSpec(getSimpleShape(shape),
                            getTensorLayout(layout, workerGrid));
}

} // namespace conversion
} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_ENABLE_OPMODEL
