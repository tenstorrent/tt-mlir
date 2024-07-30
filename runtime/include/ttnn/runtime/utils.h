// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_UTILS_H
#define TTNN_RUNTIME_UTILS_H

#include "ttmlir/Target/TTNN/Target.h"
#include "ttnn/types.hpp"

namespace tt::runtime::ttnn::utils {

inline ::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::ttnn::DataType::FLOAT32;
  // case ::tt::target::DataType::Float16:
  //   return ::ttnn::DataType::FLOAT16;
  case ::tt::target::DataType::BFloat16:
    return ::ttnn::DataType::BFLOAT16;
  case ::tt::target::DataType::UInt32:
    return ::ttnn::DataType::UINT32;
  case ::tt::target::DataType::UInt16:
    return ::ttnn::DataType::UINT16;
  // case ::tt::target::DataType::UInt8:
  //   return ::ttnn::DataType::UINT8;
  default:
    throw std::runtime_error("Unsupported data type");
  }
}

} // namespace tt::runtime::ttnn::utils

#endif
