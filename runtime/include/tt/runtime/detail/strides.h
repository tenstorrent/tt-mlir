// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_STRIDES_H
#define TT_RUNTIME_DETAIL_STRIDES_H

#include <concepts>
#include <vector>

namespace tt::runtime::common {

template <std::integral T>
inline std::vector<T> calculateStride(const std::vector<T> &shape) {
  assert(!shape.empty());
  std::vector<T> stride(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; i--) {
    stride[i - 1] = stride[i] * shape[i];
  }
  return stride;
}
} // namespace tt::runtime::common

#endif
