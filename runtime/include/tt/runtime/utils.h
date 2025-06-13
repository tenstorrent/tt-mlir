// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_UTILS_H
#define TT_RUNTIME_UTILS_H

#include <memory>
#include <type_traits>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/Common/types_generated.h"
#pragma clang diagnostic pop

namespace tt::runtime::utils {

inline std::shared_ptr<void> malloc_shared(size_t size) {
  return std::shared_ptr<void>(std::malloc(size), std::free);
}

template <typename T>
inline std::shared_ptr<void> unsafe_borrow_shared(T *ptr) {
  return std::shared_ptr<void>(static_cast<void *>(ptr), [](void *) {});
}

inline std::uint32_t dataTypeElementSize(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return 4;
  case ::tt::target::DataType::Float16:
    return 2;
  case ::tt::target::DataType::BFloat16:
    return 2;
  case ::tt::target::DataType::UInt32:
  case ::tt::target::DataType::Int32:
    return 4;
  case ::tt::target::DataType::UInt16:
    return 2;
  case ::tt::target::DataType::UInt8:
    return 1;
  default:
    assert(false && "Unsupported element size for data type");
    return 0;
  }
}

inline std::uint32_t
dataTypeElementSize(::tt::target::UnsupportedDataType dataType) {
  switch (dataType) {
  case ::tt::target::UnsupportedDataType::Int64:
    return 8;
  case ::tt::target::UnsupportedDataType::Float64:
    return 8;
  case ::tt::target::UnsupportedDataType::Bool:
    return 1;
  default:
    assert(false && "Unsupported element size for data type");
    return 0;
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
inline std::vector<uint32_t> calculateStride(const std::vector<T> &shape) {
  assert(!shape.empty());
  std::vector<uint32_t> stride(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; i--) {
    stride[i - 1] = stride[i] * shape[i];
  }
  return stride;
}

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename T>
T alignUp(T ptr, T alignment) {
  return (ptr + alignment - 1) & ~(alignment - 1);
}

template <typename dtype32, typename dtype64>
inline void handle32To64(const dtype32 *old_buffer, dtype64 *new_buffer,
                         int64_t num_elements) {

  assert(sizeof(dtype64) == 8 && "dtype64 must be 8 bytes");
  assert(sizeof(dtype32) == 4 && "dtype32 must be 4 bytes");

  for (int i = 0; i < num_elements; i++) {
    new_buffer[i] = static_cast<dtype64>(old_buffer[i]);
  }
}

template <typename dtype64, typename dtype32>
inline void handle64To32(const dtype64 *old_buffer, dtype32 *new_buffer,
                         int64_t num_elements) {

  assert(sizeof(dtype64) == 8 && "dtype64 must be 8 bytes");
  assert(sizeof(dtype32) == 4 && "dtype32 must be 4 bytes");

  for (int i = 0; i < num_elements; i++) {

    if (std::is_same_v<dtype32, int32_t> || std::is_same_v<dtype32, uint32_t>) {
      if (old_buffer[i] >
          static_cast<dtype64>(std::numeric_limits<dtype32>::max())) {
        new_buffer[i] = std::numeric_limits<dtype32>::max();
      } else if (old_buffer[i] <
                 static_cast<dtype64>(std::numeric_limits<dtype32>::lowest())) {
        new_buffer[i] = std::numeric_limits<dtype32>::lowest();
      } else {
        new_buffer[i] = static_cast<dtype32>(old_buffer[i]);
      }
    } else {
      new_buffer[i] = static_cast<dtype32>(old_buffer[i]);
    }
  }
}

inline void handleBFloat16ToBool(const uint16_t *old_buffer, bool *new_buffer,
                                 int64_t num_elements) {
  for (int i = 0; i < num_elements; i++) {
    new_buffer[i] =
        old_buffer[i] !=
        0; // 0 in bfloat16 is also 00000000 00000000, just as in uint16_t
  }
}

inline void handleBoolToBFloat16(const bool *old_buffer, uint16_t *new_buffer,
                                 int64_t num_elements) {

  assert(sizeof(bool) == 1 && "bool must be 1 byte");

  for (int i = 0; i < num_elements; i++) {
    new_buffer[i] = old_buffer[i]
                        ? 0x3f80
                        : 0; // 0x3f80 is the bfloat16 representation of 1.0
  }
}

} // namespace tt::runtime::utils

#endif
