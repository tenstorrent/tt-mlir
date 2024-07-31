// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_UTILS_H
#define TTMLIR_UTILS_H

#include <cstdint>

namespace ttmlir::utils {

template <typename Vector, typename Fn>
inline void sample(Vector const &shape, Fn fn) {
  Vector strides(shape.size());
  std::int64_t stride = 1;
  for (std::int64_t i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }

  Vector index(shape.size());
  int64_t volume = stride;
  for (int64_t i = 0; i < volume; ++i) {
    for (unsigned j = 0; j < shape.size(); ++j) {
      index[j] = (i / strides[j]) % shape[j];
    }
    fn(index);
  }
}
} // namespace ttmlir::utils

#endif
