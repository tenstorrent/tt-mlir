// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace mlir {
namespace tt {
namespace ttnn_to_emitc {

template <typename T>
struct EmitCSerializer;

template <typename T>
constexpr std::string_view EmitCSerializerV = EmitCSerializer<T>::value;

template <>
struct EmitCSerializer<int32_t> {
  static constexpr std::string_view value = "int32_t";

  static std::string convert(int32_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<int64_t> {
  static constexpr std::string_view value = "int64_t";

  static std::string convert(int64_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<uint32_t> {
  static constexpr std::string_view value = "uint32_t";

  static std::string convert(uint32_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<uint64_t> {
  static constexpr std::string_view value = "uint64_t";

  static std::string convert(uint64_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<float> {
  static constexpr std::string_view value = "float";

  static std::string convert(float value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<double> {
  static constexpr std::string_view value = "double";

  static std::string convert(double value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<bool> {
  static constexpr std::string_view value = "bool";

  static std::string convert(bool value) { return value ? "true" : "false"; }
};

template <>
struct EmitCSerializer<std::string> {
  static constexpr std::string_view value = "::std::string";

  static std::string convert(std::string value) { return "\"" + value + "\""; }
};

template <typename T, size_t k>
struct EmitCSerializer<std::array<T, k>> {
  static constexpr std::string_view value =
      "::std::array<" + EmitCSerializerV<T> + ", " + std::to_string(k) + ">";

  static std::string convert(std::array<T, k> val) {
    return std::string(EmitCSerializer<std::array<T, k>>::value);
  }
};

template <typename T>
struct EmitCSerializer<std::vector<T>> {
  static constexpr std::string_view value =
      "::std::vector<" + EmitCSerializerV<T> + ">";
};

template <typename T, typename = void>
struct EmitCTypeConverter;

template <typename T>
struct EmitCTypeConverter<T, std::enable_if_t<std::is_integral_v<T>, void>> {
  using type = T;

  static std::optional<type> convert(mlir::Attribute attr) {
    if (auto intAttr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(attr)) {
      return convert(intAttr);
    }
    return {};
  }

  static type convert(mlir::IntegerAttr attr) {
    return convert(attr.getValue());
  }

  static type convert(mlir::APInt value) {
    if constexpr (std::is_signed_v<type>) {
      return convert(value.getSExtValue());
    }
    return convert(value.getZExtValue());
  }

  template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
  static type convert(U value) {
    return static_cast<type>(value);
  }
};

template <typename T>
struct EmitCTypeConverter<T,
                          std::enable_if_t<std::is_floating_point_v<T>, void>> {
  using type = T;

  static std::optional<type> convert(mlir::Attribute attr) {
    if (auto floatAttr = mlir::dyn_cast_if_present<mlir::FloatAttr>(attr)) {
      return convert(floatAttr);
    }
    return {};
  }

  static type convert(mlir::FloatAttr attr) { return convert(attr.getValue()); }

  static type convert(mlir::APFloat value) {
    return convert(value.convertToDouble());
  }

  template <typename U,
            std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  static type convert(U value) {
    return static_cast<type>(value);
  }
};

template <typename T>
struct EmitCContainerTypeConverter {
  using type = T;
  using value_type = typename type::value_type;

  static std::optional<type> convert(mlir::Attribute attr) {
    if (auto arrayAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() ||
          EmitCTypeConverter<value_type>::convert(arrayAttr[0])) {
        return convert(arrayAttr);
      }
      return {};
    }
    if constexpr (std::is_integral_v<value_type>) {
      if (auto denseIntAttr =
              mlir::dyn_cast_if_present<mlir::DenseIntElementsAttr>(attr)) {
        return convert(denseIntAttr);
      }
    } else if constexpr (std::is_floating_point_v<value_type>) {
      if (auto denseFPAttr =
              mlir::dyn_cast_if_present<mlir::DenseFPElementsAttr>(attr)) {
        return convert(denseFPAttr);
      }
    }
    return {};
  }

  static type convert(mlir::ArrayAttr attr) {
    type result;
    for (auto element : attr) {
      result.push_back(*EmitCTypeConverter<value_type>::convert(element));
    }
    return result;
  }

  static type convert(mlir::DenseIntElementsAttr attr) {
    type result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return result;
  }

  static type convert(mlir::DenseFPElementsAttr attr) {
    type result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return result;
  }

  template <typename U>
  static type convert(llvm::ArrayRef<U> attr) {
    type result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return result;
  }
};

// TODO (azecevic): EmitCContainerTypeConverter for ttnn::SmallVector,
// ttnn::span...

template <typename T>
struct EmitCTypeConverter<std::vector<T>>
    : public EmitCContainerTypeConverter<std::vector<T>> {};

template <typename T, size_t k>
struct EmitCTypeConverter<std::array<T, k>> {
  using type = std::array<T, k>;

  static std::optional<type> convert(mlir::Attribute attr) {
    if (auto arrayAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(attr)) {
      return convert(arrayAttr);
    }
    if constexpr (std::is_integral_v<T>) {
      if (auto denseIntAttr =
              mlir::dyn_cast_if_present<mlir::DenseIntElementsAttr>(attr)) {
        return convert(denseIntAttr);
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      if (auto denseFPAttr =
              mlir::dyn_cast_if_present<mlir::DenseFPElementsAttr>(attr)) {
        return convert(denseFPAttr);
      }
    }
    return {};
  }

  static std::optional<type> convert(mlir::ArrayAttr attr) {
    if (attr.size() != k) {
      return {};
    }
    type result;
    for (size_t i = 0; i < attr.size(); ++i) {
      auto element = EmitCTypeConverter<T>::convert(attr[i]);
      if (!element) {
        return {};
      }
      result[i] = *element;
    }
    return result;
  }

  static type convert(mlir::DenseIntElementsAttr attr) {
    assert(attr.size() == k &&
           "DenseIntElementsAttr size does not match std::array size");
    type result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return result;
  }

  static type convert(mlir::DenseFPElementsAttr attr) {
    assert(attr.size() == k &&
           "DenseFPElementsAttr size does not match std::array size");
    type result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return result;
  }

  template <typename U>
  static type convert(llvm::ArrayRef<U> attr) {
    assert(attr.size() == k && "ArrayRef size does not match std::array size");
    type result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<T>::convert(element));
    }
    return result;
  }
};

template <typename First, typename... Rest>
struct EmitCTypeConverter<std::variant<First, Rest...>> {
  using type = std::variant<First, Rest...>;

  static type convert(mlir::Attribute attr) {
    if (auto tryFirst = EmitCTypeConverter<First>::convert(attr)) {
      return type(*tryFirst);
    }
    if constexpr (sizeof...(Rest) > 0) {
      auto rest = EmitCTypeConverter<std::variant<Rest...>>::convert(attr);
      return std::visit([](auto &&val) -> type { return type(val); }, rest);
    }
    llvm_unreachable("Invalid variant type");
    return {};
  }
};

template <>
struct EmitCTypeConverter<std::nullopt_t> {
  using type = std::nullopt_t;

  static type convert(mlir::Attribute attr) {
    assert(!attr && "Not an optional attribute");
    return std::nullopt;
  }

  static type convert(std::nullopt_t) { return std::nullopt; }
};

} // namespace ttnn_to_emitc
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
