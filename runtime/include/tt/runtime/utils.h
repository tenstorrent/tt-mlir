// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_UTILS_H
#define TT_RUNTIME_UTILS_H

#include <memory>

namespace tt::runtime::utils {

inline std::shared_ptr<void> malloc_shared(size_t size) {
  return std::shared_ptr<void>(std::malloc(size), std::free);
}

template <typename T>
inline std::shared_ptr<void> unsafe_borrow_shared(T *ptr) {
  return std::shared_ptr<void>(static_cast<void *>(ptr), [](void *) {});
}

} // namespace tt::runtime::utils

#endif
