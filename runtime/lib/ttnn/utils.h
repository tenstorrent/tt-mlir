// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_UTILS_H
#define TTNN_RUNTIME_UTILS_H

#include "ttmlir/Target/TTNN/Target.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::utils {

inline bool isValidTileShape(const ::tt::target::Dim2d *shape) {
  return (shape->x() == 0 and shape->y() == 0) or
         (shape->x() == 1 and shape->y() == 1) or
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

} // namespace tt::runtime::ttnn::utils

#endif
