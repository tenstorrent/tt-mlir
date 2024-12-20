// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_UTILS_H
#define TTMLIR_UTILS_H

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"

#include <cstdint>

namespace ttmlir::utils {
template <typename T>
T alignUp(T ptr, T alignment) {
  return (ptr + alignment - 1) & ~(alignment - 1);
}

template <typename T>
T alignDown(T ptr, T alignment) {
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

template <typename IntType>
IntType volume(mlir::ArrayRef<IntType> shape) {
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

template <typename Range>
std::string join(Range &&R, llvm::StringRef separator) {
  return llvm::join(
      llvm::map_range(R, [](auto &v) { return llvm::Twine(v).str(); }),
      separator);
}

// Prepacks `MlirAttribute`s stored in input array into a vector of
// `mlir::Attribute`s and then wraps that vector in `MlirAttribute` and returns
// it.
//
// This util function can be used as a helper to create an attribute from an
// array of attributes for any type defined like for example:
//
// def TT_IteratorTypeArrayAttr : TypedArrayAttrBase<TT_IteratorTypeAttr, "">;
//
// since these don't get any special Cpp class generated for them from
// tablegen.
//
// This is useful if you want to create a Pybind static geter in some attribute
// class `TT_XAttr` for which you have `TT_XArrayAttr` defined (as shown in
// example above). For example:
//
// py::class_<tt::CircularBufferAttributesAttr>(m,
//                                              "CircularBufferAttributesAttr")
//     .def_static(
//         "get",
//         [](MlirContext ctx, uint8_t cb_id, MlirAttribute core_range,
//            uint32_t total_size, uint32_t page_size, uint32_t data_format) {
//           return wrap(tt::CircularBufferAttributesAttr::get(
//               unwrap(ctx), static_cast<tt::CB>(cb_id),
//               mlir::cast<tt::CoreRangeAttr>(unwrap(core_range)), total_size,
//               page_size, static_cast<tt::DataType>(data_format)));
//         })
//     .def_static("get", [](MlirContext ctx,
//                           std::vector<MlirAttribute> attributesArray) {
//       return ::ttmlir::utils::wrapArrayOfMlirAttributesAsAttribute(ctx,
//           attributesArray);
//     });
inline MlirAttribute wrapArrayOfMlirAttributesAsAttribute(
    MlirContext ctx, std::vector<MlirAttribute> &attributesArray) {
  std::vector<mlir::Attribute> unwrappedAttributesArray;
  for (auto attr : attributesArray) {
    unwrappedAttributesArray.push_back(unwrap(attr));
  }
  return wrap(mlir::ArrayAttr::get(unwrap(ctx), unwrappedAttributesArray));
}

// Checks if the type of the given `mlir::Value` is a ranked tensor type.
inline bool isRankedTensor(mlir::Value v) {
  return mlir::isa<mlir::RankedTensorType>(v.getType());
}

// Returns the element received as a parameter. Useful as a callback for
// higher-order functions.
template <typename T>
inline T identity(T x) {
  return x;
}

// Returns a vector of indices `permutation` such that input[permutation[i]] ==
// output[i], for all i. Assumes that input and output have the same elements.
// Example:  input = [1, 2, 3], output = [3, 1, 2] -> [2, 0, 1]
template <typename T>
inline llvm::SmallVector<int64_t>
generatePermutation(llvm::ArrayRef<T> input, llvm::ArrayRef<T> output) {
  assert(input.size() == output.size());

  llvm::DenseMap<T, int64_t> indices;
  for (const auto [index, value] : llvm::enumerate(input)) {
    indices[value] = index;
  }
  llvm::SmallVector<int64_t> permutation;
  for (const T &dim : output) {
    permutation.push_back(indices[dim]);
  }
  return permutation;
}

// Returns a vector `output`, such that output[i] = input[permutation[i]], for
// all i. Assumes that permutation is a valid permutation of the indices of
// input. Example:  input = [1, 2, 3], permutation = [2, 0, 1] -> [3, 1, 2]
template <typename T>
inline llvm::SmallVector<T>
applyPermutation(llvm::ArrayRef<T> input, llvm::ArrayRef<int64_t> permutation) {
  assert(input.size() == permutation.size());

  llvm::SmallVector<T> output(input.size());

  llvm::transform(permutation, output.begin(),
                  [&](const int64_t i) { return input[i]; });

  return output;
}

// Returns a vector `inversePermutation`, such that
// inversePermutation[permutation[i]] = i, for all i. Assumes that permutation
// is a valid permutation of a range(0, permutation.size()). Example:
// permutation = [2, 0, 1] -> [1, 2, 0]
inline llvm::SmallVector<int64_t>
inversePermutation(llvm::ArrayRef<int64_t> permutation) {
  llvm::SmallVector<int64_t> inversePermutation(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    inversePermutation[permutation[i]] = i;
  }
  return inversePermutation;
}

} // namespace ttmlir::utils

#endif
