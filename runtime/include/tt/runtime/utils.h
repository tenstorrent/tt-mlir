// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_UTILS_H
#define TT_RUNTIME_UTILS_H

#include <memory>

#include "ttmlir/Target/Common/types_generated.h"

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

} // namespace tt::runtime::utils

#endif
