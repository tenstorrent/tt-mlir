// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// This namespace contains mock declarations of TTNN types for the purpose of
// conversion.
namespace ttnn {
template <typename T>
class SmallVector;
} // namespace ttnn

namespace mlir {
namespace tt {
namespace ttnn_to_emitc {

template <typename T>
struct TypeName;

template <typename T>
const std::string TypeNameV = TypeName<T>::value;

template <>
struct TypeName<int32_t> {
  inline static const std::string value = "int32_t";
};

template <>
struct TypeName<int64_t> {
  inline static const std::string value = "int64_t";
};

template <>
struct TypeName<uint32_t> {
  inline static const std::string value = "uint32_t";
};

template <>
struct TypeName<uint64_t> {
  inline static const std::string value = "uint64_t";
};

template <>
struct TypeName<float> {
  inline static const std::string value = "float";
};

template <>
struct TypeName<double> {
  inline static const std::string value = "double";
};

template <>
struct TypeName<bool> {
  inline static const std::string value = "bool";
};

template <>
struct TypeName<std::string> {
  inline static const std::string value = "::std::string";
};

template <typename T, size_t k>
struct TypeName<std::array<T, k>> {
  inline static const std::string value =
      "::std::array<" + TypeNameV<T> + ", " + std::to_string(k) + ">";
};

template <typename T>
struct TypeName<std::vector<T>> {
  inline static const std::string value = "::std::vector<" + TypeNameV<T> + ">";
};

template <typename T>
struct TypeName<::ttnn::SmallVector<T>> {
  inline static const std::string value =
      "::ttnn::SmallVector<" + TypeNameV<T> + ">";
};

template <typename T, typename Enable = void>
struct EmitCTypeConverter;

// Converter for integral types.
template <typename T>
struct EmitCTypeConverter<T, std::enable_if_t<std::is_integral_v<T>, void>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto intAttr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(attr)) {
      return convert(intAttr);
    }
    return {};
  }

  static std::string convert(mlir::IntegerAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(mlir::APInt value) {
    if constexpr (std::is_signed_v<T>) {
      return convert(value.getSExtValue());
    }
    return convert(value.getZExtValue());
  }

  template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
  static std::string convert(U value) {
    return std::to_string(static_cast<T>(value));
  }
};

// Converter for floating point types.
template <typename T>
struct EmitCTypeConverter<T,
                          std::enable_if_t<std::is_floating_point_v<T>, void>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto floatAttr = mlir::dyn_cast_if_present<mlir::FloatAttr>(attr)) {
      return convert(floatAttr);
    }
    return {};
  }

  static std::string convert(mlir::FloatAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(mlir::APFloat value) {
    return convert(value.convertToDouble());
  }

  template <typename U,
            std::enable_if_t<std::is_floating_point_v<U>, bool> = true>
  static std::string convert(U value) {
    return std::to_string(static_cast<T>(value));
  }
};

// Convert container types (std::vector, ttnn::SmallVector, etc.).
template <typename T>
struct EmitCContainerTypeConverter {
  using value_type = typename T::value_type;

  static std::optional<std::string> convert(mlir::Attribute attr) {
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

  static std::string convert(mlir::ArrayAttr attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(*EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseFPElementsAttr attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::string convert(llvm::ArrayRef<U> attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

private:
  static std::string convert(const std::vector<std::string> &values) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<T> << "{";
    llvm::interleaveComma(values, rso);
    rso << "}";
    return buf;
  }
};

template <typename T>
struct EmitCTypeConverter<std::vector<T>>
    : public EmitCContainerTypeConverter<std::vector<T>> {};

template <typename T>
struct EmitCTypeConverter<::ttnn::SmallVector<T>>
    : public EmitCContainerTypeConverter<::ttnn::SmallVector<T>> {};

template <typename T, size_t k>
struct EmitCTypeConverter<std::array<T, k>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
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

  static std::optional<std::string> convert(mlir::ArrayAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    ;
    for (size_t i = 0; i < attr.size(); ++i) {
      auto element = EmitCTypeConverter<T>::convert(attr[i]);
      if (!element) {
        return {};
      }
      result[i] = *element;
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    assert(attr.size() == k &&
           "DenseIntElementsAttr size does not match std::array size");
    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseFPElementsAttr attr) {
    assert(attr.size() == k &&
           "DenseFPElementsAttr size does not match std::array size");
    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  template <typename U>
  static std::string convert(llvm::ArrayRef<U> attr) {
    assert(attr.size() == k && "ArrayRef size does not match std::array size");
    std::array<std::string, k> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<T>::convert(element));
    }
    return convert(result);
  }

private:
  static std::string convert(const std::array<std::string, k> &values) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<std::array<T, k>> << "{";
    llvm::interleaveComma(values, rso);
    rso << "}";
    return buf;
  }
};

template <typename First, typename... Rest>
struct EmitCTypeConverter<std::variant<First, Rest...>> {
  static std::string convert(mlir::Attribute attr) {
    if (auto tryFirst = EmitCTypeConverter<First>::convert(attr)) {
      return *tryFirst;
    }

    if constexpr (sizeof...(Rest) > 0) {
      return EmitCTypeConverter<std::variant<Rest...>>::convert(attr);
    }

    llvm_unreachable("Invalid variant type");
  }
};

inline std::string convert(ttnn::ShapeAttr attr) {
  if (!attr) {
    return "::std::nullopt";
  }

  std::string buf;
  llvm::raw_string_ostream rso(buf);

  auto shape = attr.getShape();
  rso << "::ttnn::Shape({";
  llvm::interleaveComma(shape, rso);
  rso << "})";

  return buf;
}

inline std::string convert(tt::DataTypeAttr attr) {
  if (!attr) {
    return "::std::nullopt";
  }

  switch (attr.getValue()) {
  case tt::DataType::BFloat16:
    return "::ttnn::DataType::BFLOAT16";
  case tt::DataType::Float32:
    return "::ttnn::DataType::FLOAT32";
  case tt::DataType::UInt32:
    return "::ttnn::DataType::UINT32";
  case tt::DataType::BFP_BFloat8:
    return "::ttnn::DataType::BFLOAT8_B";
  case tt::DataType::BFP_BFloat4:
    return "::ttnn::DataType::BFLOAT4_B";
  case tt::DataType::UInt8:
    return "::ttnn::DataType::UINT8";
  case tt::DataType::UInt16:
    return "::ttnn::DataType::UINT16";
  // TODO(svuckovic):
  // Add support for INT32
  //
  // case tt::DataType::Int32:
  //   return builder.getType<emitc::OpaqueAttr>("ttnn::DataType::INT32");
  case tt::DataType::Float16:
  case tt::DataType::BFP_Float2:
  case tt::DataType::BFP_Float4:
  case tt::DataType::BFP_Float8:
  case tt::DataType::BFP_BFloat2:
    llvm_unreachable("Unsupported ttnn::DataType");
  }

  llvm_unreachable("Unkonwn tt::DataType");
}

inline std::string convert(ttnn::LayoutAttr attr) {
  if (!attr) {
    return "::std::nullopt";
  }

  switch (attr.getValue()) {
  case ttnn::Layout::RowMajor:
    return "::ttnn::Layout::ROW_MAJOR";
  case ttnn::Layout::Tile:
    return "::ttnn::Layout::TILE";
  case ttnn::Layout::Invalid:
    return "::ttnn::Layout::INVALID";
  }

  llvm_unreachable("Unknown ttnn::Layout");
}

inline std::string convert(ttnn::TensorMemoryLayoutAttr attr) {
  if (!attr) {
    return "::std::nullopt";
  }

  switch (attr.getValue()) {
  case ttnn::TensorMemoryLayout::BlockSharded:
    return "::ttnn::TensorMemoryLayout::BLOCK_SHARDED";
  case ttnn::TensorMemoryLayout::HeightSharded:
    return "::ttnn::TensorMemoryLayout::HEIGHT_SHARDED";
  case ttnn::TensorMemoryLayout::Interleaved:
    return "::ttnn::TensorMemoryLayout::INTERLEAVED";
  case ttnn::TensorMemoryLayout::SingleBank:
    return "::ttnn::TensorMemoryLayout::SINGLE_BANK";
  case ttnn::TensorMemoryLayout::WidthSharded:
    return "::ttnn::TensorMemoryLayout::WIDTH_SHARDED";
  }

  llvm_unreachable("Unknown ttnn::TensorMemoryLayout");
}

inline std::string convert(ttnn::BufferTypeAttr attr) {
  if (!attr) {
    return "::std::nullopt";
  }

  switch (attr.getValue()) {
  case ttnn::BufferType::DRAM:
    return "::ttnn::BufferType::DRAM";
  case ttnn::BufferType::L1:
    return "::ttnn::BufferType::L1";
  case ttnn::BufferType::L1Small:
    return "::ttnn::BufferType::L1_SMALL";
  case ttnn::BufferType::SystemMemory:
    return "::ttnn::BufferType::SYSTEM_MEMORY";
  case ttnn::BufferType::Trace:
    return "::ttnn::BufferType::TRACE";
  }

  llvm_unreachable("Unknown ttnn::BufferType");
}

inline std::string convert(ttnn::MemoryConfigAttr attr) {
  if (!attr) {
    return "::std::nullopt";
  }

  // TODO (azecevic): Add ShardSpec once it's modeled in the `MemoryConfigAttr`.
  std::string buf;
  llvm::raw_string_ostream rso(buf);
  rso << "::ttnn::MemoryConfig(";
  rso << convert(attr.getTensorMemoryLayout()) << ", ";
  rso << convert(attr.getBufferType());
  rso << ")";
  return buf;
}

} // namespace ttnn_to_emitc
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
