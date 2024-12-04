// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_UTILS_H
#define TTMLIR_UTILS_H

#include <cstdint>
#include <sstream>

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

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

template <typename T>
std::string join(const llvm::SmallVector<T> &vec,
                 const std::string &delimiter) {
  std::ostringstream result;
  for (size_t i = 0; i < vec.size(); ++i) {
    result << vec[i];
    if (i != vec.size() - 1) {
      result << delimiter;
    }
  }
  return result.str();
}

// Prepacks `MlirAttribute`s stored in input array into a vector of
// `mlir::Attribute`s and then wraps that vector in `MlirAttribute` and returns
// it.
//
// This util function can be used as a helper to create an attribute from an
// array of attributes for any type defined like for example:
//
// `def TT_OperandConstraintArrayAttr
//            : TypedArrayAttrBase<TT_OperandConstraintAttr, "">;`
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

} // namespace ttmlir::utils

#endif
