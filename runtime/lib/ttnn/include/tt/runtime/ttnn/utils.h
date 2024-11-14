// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_UTILS_H
#define TTNN_RUNTIME_UTILS_H

#include "flatbuffers/vector.h"
#include "ttmlir/Target/Common/types_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::utils {

inline bool isValidTileShape(const ::tt::target::Dim2d *shape) {
  return (shape->x() == 1 and shape->y() == 1) or
         (shape->x() == 32 and shape->y() == 32);
}

inline ::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::ttnn::DataType::FLOAT32;
  case ::tt::target::DataType::BFloat16:
    return ::ttnn::DataType::BFLOAT16;
  case ::tt::target::DataType::BFP_BFloat8:
    return ::ttnn::DataType::BFLOAT8_B;
  case ::tt::target::DataType::BFP_BFloat4:
    return ::ttnn::DataType::BFLOAT4_B;
  case ::tt::target::DataType::UInt32:
    return ::ttnn::DataType::UINT32;
  case ::tt::target::DataType::UInt16:
    return ::ttnn::DataType::UINT16;

  default:
    throw std::runtime_error("Unsupported data type");
  }
}

inline ::tt::target::DataType fromTTNNDataType(::ttnn::DataType dataType) {
  switch (dataType) {
  case ::ttnn::DataType::FLOAT32:
    return ::tt::target::DataType::Float32;
  case ::ttnn::DataType::BFLOAT16:
    return ::tt::target::DataType::BFloat16;
  case ::ttnn::DataType::BFLOAT8_B:
    return ::tt::target::DataType::BFP_BFloat8;
  case ::ttnn::DataType::BFLOAT4_B:
    return ::tt::target::DataType::BFP_BFloat4;
  case ::ttnn::DataType::UINT32:
    return ::tt::target::DataType::UInt32;
  case ::ttnn::DataType::UINT16:
    return ::tt::target::DataType::UInt16;

  default:
    throw std::runtime_error("Unsupported data type");
  }
}

inline ::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout) {
  switch (layout) {
  case ::tt::target::TensorLayout::Tile:
    return ::ttnn::Layout::TILE;
  case ::tt::target::TensorLayout::RowMajor:
    return ::ttnn::Layout::ROW_MAJOR;
  default:
    throw std::runtime_error("Unsupported layout");
  }
}

inline ::ttnn::TensorMemoryLayout
toTTNNTensorMemoryLayout(::tt::target::TensorMemoryLayout tensorMemoryLayout) {

  switch (tensorMemoryLayout) {
  case ::tt::target::TensorMemoryLayout::Interleaved:
    return ::ttnn::TensorMemoryLayout::INTERLEAVED;
  case ::tt::target::TensorMemoryLayout::SingleBank:
    return ::ttnn::TensorMemoryLayout::SINGLE_BANK;
  case ::tt::target::TensorMemoryLayout::HeightSharded:
    return ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
  case ::tt::target::TensorMemoryLayout::WidthSharded:
    return ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
  case ::tt::target::TensorMemoryLayout::BlockSharded:
    return ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
  case ::tt::target::TensorMemoryLayout::None:
    assert(false &&
           "Unsupported tensor memory layout TensorMemoryLayout::None");
  }
}

// This method will be deprecated in favor of method below
//
inline ::tt::tt_metal::BufferType
toTTNNBufferType(::tt::target::MemorySpace memorySpace) {
  switch (memorySpace) {
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO:
    return ::tt::tt_metal::BufferType::SYSTEM_MEMORY;
  case ::tt::target::MemorySpace::DeviceDRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case ::tt::target::MemorySpace::DeviceL1:
    return ::tt::tt_metal::BufferType::L1;
  }
}

// Prefer to use this method
//
inline ::ttnn::BufferType
toTTNNBufferType(::tt::target::BufferType bufferType) {

  switch (bufferType) {
  case ::tt::target::BufferType::DRAM:
    return ::ttnn::BufferType::DRAM;
  case ::tt::target::BufferType::L1:
    return ::ttnn::BufferType::L1;
  case ::tt::target::BufferType::SystemMemory:
    return ::ttnn::BufferType::SYSTEM_MEMORY;
  case ::tt::target::BufferType::L1Small:
    return ::ttnn::BufferType::L1_SMALL;
  case ::tt::target::BufferType::Trace:
    return ::ttnn::BufferType::TRACE;
  }
};

inline std::vector<uint32_t>
toShapeFromFBShape(const flatbuffers::Vector<int32_t> &vec) {
  return std::vector<uint32_t>(vec.begin(), vec.end());
}

} // namespace tt::runtime::ttnn::utils

#endif
