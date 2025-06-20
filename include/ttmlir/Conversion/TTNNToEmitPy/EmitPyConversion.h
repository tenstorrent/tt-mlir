// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H

#include <string>
#include <vector>

namespace ttnn {

struct Tensor;

} // namespace ttnn

namespace mlir {
namespace tt {
namespace ttnn_to_emitpy {

template <typename T>
struct TypeName;

template <typename T>
const std::string TypeNameV = TypeName<T>::value;

template <>
struct TypeName<::ttnn::Tensor> {
  inline static const std::string value = "ttnn.Tensor";
};

template <typename T>
struct TypeName<std::vector<T>> {
  inline static const std::string value = "vector<" + TypeNameV<T> + ">";
};
} // namespace ttnn_to_emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H
