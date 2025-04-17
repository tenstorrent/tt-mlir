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

#include "tt/runtime/detail/flatbuffer_operator_ostream.h"

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

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
inline std::vector<uint32_t> calculateStride(const std::vector<T> &shape) {
  assert(!shape.empty());
  std::vector<uint32_t> stride(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; i--) {
    stride[i - 1] = stride[i] * shape[i];
  }
  return stride;
}

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

} // namespace tt::runtime::utils

#endif
