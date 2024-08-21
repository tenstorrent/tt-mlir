// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_UTILS_H
#define TTNN_RUNTIME_UTILS_H

#include "common/base_types.hpp"
#include "flatbuffers/vector.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttnn/types.hpp"
#include "types_generated.h"

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

inline std::vector<uint32_t>
toShapeFromFBShape(const flatbuffers::Vector<int32_t> &vec) {
  return std::vector<uint32_t>(vec.begin(), vec.end());
}

inline CoreRangeSet get_core_range_set(const target::Dim2dRange *core_spec) {
  uint32_t x_start = core_spec->loc().x();
  uint32_t y_start = core_spec->loc().y();
  uint32_t x_size = core_spec->size().x();
  uint32_t y_size = core_spec->size().x();
  CoreRange cr({x_start, y_start}, {x_start + x_size - 1, y_start + y_size - 1});
  CoreRangeSet crs({cr});
  return crs;
}

inline MathFidelity toTTNNMathFidelity(target::MathFidelity fbs_math_fidelity) {
  switch(fbs_math_fidelity) {
    case target::MathFidelity::LoFi:
      return MathFidelity::LoFi;
    case target::MathFidelity::HiFi2:
      return MathFidelity::HiFi2;
    case target::MathFidelity::HiFi3:
      return MathFidelity::HiFi3;
    case target::MathFidelity::HiFi4:
      return MathFidelity::HiFi4;
    case target::MathFidelity::Invalid:
      return MathFidelity::Invalid;
  }

  return MathFidelity::Invalid;
}

} // namespace tt::runtime::ttnn::utils

#endif
