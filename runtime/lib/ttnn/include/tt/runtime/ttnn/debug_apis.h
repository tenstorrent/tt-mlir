// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_DEBUG_APIS_H
#define TT_RUNTIME_TTNN_DEBUG_APIS_H

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "ttmlir/Target/TTNN/Target.h"

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
#define RUNTIME_DEBUG_MAYBE_CONST_INLINE
#else
#define RUNTIME_DEBUG_MAYBE_CONST_INLINE                                       \
  inline __attribute__((always_inline, const))
#endif

namespace tt::runtime::ttnn::debug {

inline std::string toString(const ::ttnn::Layout &layout) {
  switch (layout) {
  case ::ttnn::Layout::ROW_MAJOR:
    return "ROW_MAJOR";
  case ::ttnn::Layout::TILE:
    return "TILE";
  case ::ttnn::Layout::INVALID:
    return "INVALID";
  }
}

inline std::string toString(const ::ttnn::DataType &dtype) {
  switch (dtype) {
  case ::ttnn::DataType::FLOAT32:
    return "FLOAT32";
  case ::ttnn::DataType::BFLOAT16:
    return "BFLOAT16";
  case ::ttnn::DataType::BFLOAT8_B:
    return "BFLOAT8_B";
  case ::ttnn::DataType::BFLOAT4_B:
    return "BFLOAT4_B";
  case ::ttnn::DataType::UINT32:
    return "UINT32";
  case ::ttnn::DataType::UINT16:
    return "UINT16";
  case ::ttnn::DataType::UINT8:
    return "UINT8";
  case ::ttnn::DataType::INT32:
    return "INT32";
  case ::ttnn::DataType::INVALID:
    return "INVALID";
  }
}

inline std::string toString(const ::ttnn::StorageType &storageType) {
  switch (storageType) {
  case ::ttnn::StorageType::HOST:
    return "HOST";
  case ::ttnn::StorageType::DEVICE:
    return "DEVICE";
  case ::ttnn::StorageType::MULTI_DEVICE_HOST:
    return "MULTI_DEVICE_HOST";
  default:
    return "UNKNOWN";
  }
}

RUNTIME_DEBUG_MAYBE_CONST_INLINE void
checkTensorRefMatchesTTNNTensor(const ::tt::target::ttnn::TensorRef *tensorRef,
                                const ::ttnn::Tensor &ttnnTensor)
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
    ;
#else
{
}
#endif

#undef RUNTIME_DEBUG_MAYBE_CONST_INLINE

} // namespace tt::runtime::ttnn::debug

#endif // TT_RUNTIME_TTNN_DEBUG_APIS_H
