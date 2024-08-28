// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_UTILS_H
#define TTNN_RUNTIME_UTILS_H

#include "flatbuffers/vector.h"
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
    
inline ::tt::tt_metal::TensorMemoryLayout
toTensorMemoryLayout(::tt::target::TensorMemoryLayout memLayout) {
  switch (memLayout) {
  case ::tt::target::TensorMemoryLayout::Interleaved:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  case ::tt::target::TensorMemoryLayout::SingleBank:
    return ::tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
  case ::tt::target::TensorMemoryLayout::HeightSharded:
    return ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
  case ::tt::target::TensorMemoryLayout::WidthSharded:
    return ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
  case ::tt::target::TensorMemoryLayout::BlockSharded:
    return ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;

  default:
    throw std::runtime_error("Unsupported shard strategy");
  }
}

inline std::vector<uint32_t>
toShapeFromFBShape(const flatbuffers::Vector<int32_t> &vec) {
  return std::vector<uint32_t>(vec.begin(), vec.end());
}

} // namespace tt::runtime::ttnn::utils

#endif
