// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_UTILS_H
#define TTMLIR_UTILS_H

#include <cstdint>

#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

namespace ttmlir::utils {
template <typename T> T alignUp(T ptr, T alignment) {
  return (ptr + alignment - 1) & ~(alignment - 1);
}

template <typename T> T alignDown(T ptr, T alignment) {
  return ptr & ~(alignment - 1);
}

template <typename Vector, typename Fn>
inline void sample(Vector const &shape, Fn fn) {
  llvm::SmallVector<std::int64_t, 8> strides(shape.size());
  std::int64_t stride = 1;
  for (std::int64_t i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }

  llvm::SmallVector<std::int64_t, 8> index(shape.size());
  int64_t volume = stride;
  for (int64_t i = 0; i < volume; ++i) {
    for (unsigned j = 0; j < shape.size(); ++j) {
      index[j] = (i / strides[j]) % shape[j];
    }
    fn(index);
  }
}

template <typename Vector>
llvm::SmallVector<int64_t> evalShape(mlir::AffineMap map, Vector shape) {
  mlir::SmallVector<int64_t> lastIndex;
  for (auto dim : shape) {
    lastIndex.push_back(dim - 1);
  }

  auto result = map.compose(lastIndex);
  for (auto &dim : result) {
    dim += 1;
  }
  return result;
}

template <typename IntType> IntType volume(mlir::ArrayRef<IntType> shape) {
  IntType result = 1;
  for (auto dim : shape) {
    result *= dim;
  }
  return result;
}

template <typename Enum>
constexpr std::underlying_type_t<Enum> enum_as_int(Enum e) {
  return static_cast<std::underlying_type_t<Enum>>(e);
}
} // namespace ttmlir::utils

#endif
