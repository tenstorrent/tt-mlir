// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H

#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

// This namespace contains mock definitions of TTNN types for the purpose of
// conversion.
namespace ttsl {
template <typename T>
struct SmallVector {
  using value_type = T;
};
} // namespace ttsl

namespace ttnn {
struct Shape;

struct ShardSpec;
struct CoreRangeSet;
struct CoreRange;
struct CoreCoord;

struct DataType;
struct TensorMemoryLayout;
struct Layout;
struct MemoryConfig;
struct BufferType;

namespace types {
struct ShardOrientation;
struct ShardMode;
} // namespace types

struct Tensor;
} // namespace ttnn

namespace mlir {
namespace tt {
namespace ttnn_to_emitpy {

template <typename T, typename Enable = void>
struct TypeName;

template <typename T>
const std::string TypeNameV = TypeName<T>::value;

template <typename T>
struct is_int_type : public std::false_type {};

template <>
struct is_int_type<int32_t> : public std::true_type {};

template <>
struct is_int_type<int64_t> : public std::true_type {};

template <>
struct is_int_type<uint32_t> : public std::true_type {};

template <>
struct is_int_type<uint64_t> : public std::true_type {};

template <typename T>
inline constexpr bool is_int_type_v = is_int_type<T>::value;

template <typename T>
struct TypeName<T, std::enable_if_t<is_int_type_v<T>, void>> {
  inline static const std::string value = "int";
};

template <typename T>
struct TypeName<T, std::enable_if_t<std::is_floating_point_v<T>, void>> {
  inline static const std::string value = "float";
};

template <>
struct TypeName<bool> {
  inline static const std::string value = "bool";
};

template <>
struct TypeName<::ttnn::Tensor> {
  inline static const std::string value = "ttnn.Tensor";
};

template <typename T>
struct is_list_type : public std::false_type {};

template <typename T>
struct is_list_type<std::vector<T>> : public std::true_type {};

template <typename T>
struct is_list_type<::ttsl::SmallVector<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_list_type_v = is_list_type<T>::value;

template <typename T>
struct TypeName<T, std::enable_if_t<is_list_type_v<T>, void>> {
  using value_type = typename T::value_type;
  inline static const std::string value = "list<" + TypeNameV<value_type> + ">";
};

template <typename T, size_t k>
struct TypeName<std::array<T, k>> {
  inline static const std::string value =
      "list<" + TypeNameV<T> + ", " + std::to_string(k) + ">";
};

template <>
struct TypeName<std::string> {
  inline static const std::string value = "str";
};

template <>
struct TypeName<std::nullopt_t> {
  // This is a special case, as std::nullopt is not a type, but is the only
  // value of type std::nullopt_t.
  inline static const std::string value = "None";
};

template <typename T>
struct TypeName<std::set<T>> {
  inline static const std::string value = "set<" + TypeNameV<T> + ">";
};

template <>
struct TypeName<::ttnn::CoreCoord> {
  inline static const std::string value = "ttnn.CoreCoord";
};

template <>
struct TypeName<::ttnn::CoreRange> {
  inline static const std::string value = "ttnn.CoreRange";
};

template <>
struct TypeName<::ttnn::CoreRangeSet> {
  inline static const std::string value = "ttnn.CoreRangeSet";
};

template <>
struct TypeName<::ttnn::ShardSpec> {
  inline static const std::string value = "ttnn.ShardSpec";
};

template <>
struct TypeName<::ttnn::types::ShardOrientation> {
  inline static const std::string value = "ttnn.ShardOrientation";
};

template <>
struct TypeName<::ttnn::types::ShardMode> {
  inline static const std::string value = "ttnn.ShardMode";
};

template <>
struct TypeName<::ttnn::DataType> {
  inline static const std::string value = "ttnn.DataType";
};

template <>
struct TypeName<::ttnn::TensorMemoryLayout> {
  inline static const std::string value = "ttnn.TensorMemoryLayout";
};

template <>
struct TypeName<::ttnn::Layout> {
  inline static const std::string value = "ttnn.Layout";
};

template <>
struct TypeName<::ttnn::MemoryConfig> {
  inline static const std::string value = "ttnn.MemoryConfig";
};

template <>
struct TypeName<::ttnn::BufferType> {
  inline static const std::string value = "ttnn.BufferType";
};

template <>
struct TypeName<::ttnn::Shape> {
  inline static const std::string value = "ttnn.Shape";
};

template <typename T, typename Enable = void>
struct EmitPyTypeConverter;

template <>
struct EmitPyTypeConverter<bool> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto boolAttr = mlir::dyn_cast_if_present<mlir::BoolAttr>(attr)) {
      return convert(boolAttr);
    }
    return {};
  }

  static std::string convert(mlir::APInt attr) {
    assert(attr.getBitWidth() == 1 && "Expected a 1-bit APInt");
    return convert(static_cast<bool>(attr.getZExtValue()));
  }

  static std::string convert(mlir::BoolAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(bool value) { return value ? "True" : "False"; }
};

template <>
struct EmitPyTypeConverter<std::string> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto strAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(attr)) {
      return convert(strAttr);
    }
    return {};
  }

  static std::string convert(mlir::StringAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(mlir::StringRef attr) {
    return convert(attr.str());
  }

  static std::string convert(std::string value) { return "\"" + value + "\""; }
};

// Converter for integral types.
template <typename T>
struct EmitPyTypeConverter<T, std::enable_if_t<std::is_integral_v<T>, void>> {
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

  template <typename U>
  static std::enable_if_t<std::is_integral_v<U>, std::string> convert(U value) {
    return std::to_string(static_cast<T>(value));
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::CoreCoord> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto coreCoordAttr =
            mlir::dyn_cast_if_present<ttnn::CoreCoordAttr>(attr)) {
      return convert(coreCoordAttr);
    }
    return {};
  }

  static std::string convert(ttnn::CoreCoordAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::CoreCoord>;
    rso << "[";
    rso << EmitPyTypeConverter<size_t>::convert(attr.getX()) << ", ";
    rso << EmitPyTypeConverter<size_t>::convert(attr.getY());
    rso << "]";

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::CoreRange> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto coreRangeAttr =
            mlir::dyn_cast_if_present<ttnn::CoreRangeAttr>(attr)) {
      return convert(coreRangeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::CoreRangeAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::CoreRange>;
    rso << "[";
    rso << EmitPyTypeConverter<::ttnn::CoreCoord>::convert(attr.getStartCoord())
        << ", ";
    rso << EmitPyTypeConverter<::ttnn::CoreCoord>::convert(attr.getEndCoord());
    rso << "]";

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::types::ShardOrientation> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardOrientationAttr =
            mlir::dyn_cast_if_present<ttnn::ShardOrientationAttr>(attr)) {
      return convert(shardOrientationAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardOrientationAttr attr) {
    assert(
        attr &&
        "expected non-null attribute, call "
        "EmitPyTypeConverter<std::optional<::ttnn::types::ShardOrientation>>:"
        ":convert(attr) if attribute is optional");
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::ShardOrientation attr) {
    switch (attr) {
    case ttnn::ShardOrientation::RowMajor:
      return TypeNameV<::ttnn::types::ShardOrientation> + ".ROW_MAJOR";
    case ttnn::ShardOrientation::ColMajor:
      return TypeNameV<::ttnn::types::ShardOrientation> + ".COL_MAJOR";
    }

    llvm_unreachable("Unknown ttnn.ShardOrientation");
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::types::ShardMode> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardModeAttr =
            mlir::dyn_cast_if_present<ttnn::ShardModeAttr>(attr)) {
      return convert(shardModeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardModeAttr attr) {
    assert(attr &&
           "expected non-null attribute, call "
           "EmitPyTypeConverter<std::optional<::ttnn::types::ShardMode>>"
           "::convert(attr) if attribute is optional");
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::ShardMode attr) {
    switch (attr) {
    case ttnn::ShardMode::Physical:
      return TypeNameV<::ttnn::types::ShardMode> + ".PHYSICAL";
    case ttnn::ShardMode::Logical:
      return TypeNameV<::ttnn::types::ShardMode> + ".LOGICAL";
    }

    llvm_unreachable("Unknown ttnn.ShardMode");
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::DataType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto dataTypeAttr =
            mlir::dyn_cast_if_present<ttcore::DataTypeAttr>(attr)) {
      return convert(dataTypeAttr);
    }
    return {};
  }

  static std::string convert(ttcore::DataType attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::DataType> << ".";
    switch (attr) {
    case ttcore::DataType::BFloat16:
      rso << "BFLOAT16";
      break;
    case ttcore::DataType::Float32:
      rso << "FLOAT32";
      break;
    case ttcore::DataType::UInt32:
      rso << "UINT32";
      break;
    case ttcore::DataType::BFP_BFloat8:
      rso << "BFLOAT8_B";
      break;
    case ttcore::DataType::BFP_BFloat4:
      rso << "BFLOAT4_B";
      break;
    case ttcore::DataType::UInt8:
      rso << "UINT8";
      break;
    case ttcore::DataType::UInt16:
      rso << "UINT16";
      break;
    case ttcore::DataType::Int32:
      rso << "INT32";
      break;
    case ttcore::DataType::Float16:
    case ttcore::DataType::BFP_Float2:
    case ttcore::DataType::BFP_Float4:
    case ttcore::DataType::BFP_Float8:
    case ttcore::DataType::BFP_BFloat2:
      llvm_unreachable("Unsupported ttnn.DataType");
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::TensorMemoryLayout> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto tensorMemoryLayoutAttr =
            mlir::dyn_cast_if_present<ttnn::TensorMemoryLayoutAttr>(attr)) {
      return convert(tensorMemoryLayoutAttr);
    }
    return {};
  }

  static std::string convert(ttnn::TensorMemoryLayoutAttr attr) {
    if (!attr) {
      return convert(ttnn::TensorMemoryLayout::Interleaved);
    }
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::TensorMemoryLayout attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::TensorMemoryLayout> << ".";
    switch (attr) {
    case ttnn::TensorMemoryLayout::BlockSharded:
      rso << "BLOCK_SHARDED";
      break;
    case ttnn::TensorMemoryLayout::HeightSharded:
      rso << "HEIGHT_SHARDED";
      break;
    case ttnn::TensorMemoryLayout::Interleaved:
      rso << "INTERLEAVED";
      break;
    case ttnn::TensorMemoryLayout::WidthSharded:
      rso << "WIDTH_SHARDED";
      break;
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::Layout> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto layoutAttr = mlir::dyn_cast_if_present<ttnn::LayoutAttr>(attr)) {
      return convert(layoutAttr);
    }
    return {};
  }

  static std::string convert(ttnn::LayoutAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::Layout attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::Layout> << ".";
    switch (attr) {
    case ttnn::Layout::RowMajor:
      rso << "ROW_MAJOR";
      break;
    case ttnn::Layout::Tile:
      rso << "TILE";
      break;
    case ttnn::Layout::Invalid:
      rso << "INVALID";
      break;
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::BufferType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto bufferTypeAttr =
            mlir::dyn_cast_if_present<ttnn::BufferTypeAttr>(attr)) {
      return convert(bufferTypeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::BufferTypeAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::BufferType attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::BufferType> << ".";
    switch (attr) {
    case ttnn::BufferType::DRAM:
      rso << "DRAM";
      break;
    case ttnn::BufferType::L1:
      rso << "L1";
      break;
    case ttnn::BufferType::L1Small:
      rso << "L1_SMALL";
      break;
    case ttnn::BufferType::SystemMemory:
      rso << "SYSTEM_MEMORY";
      break;
    case ttnn::BufferType::Trace:
      rso << "TRACE";
      break;
    }

    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::Shape> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shapeAttr = mlir::dyn_cast_if_present<ttnn::ShapeAttr>(attr)) {
      return convert(shapeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShapeAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);

    auto shape = attr.getShape();
    rso << TypeNameV<::ttnn::Shape> << "([";
    llvm::interleaveComma(shape, rso);
    rso << "])";

    return buf;
  }
};

// Convert container types (std::vector, ttnn::SmallVector, etc.).
template <typename T>
struct EmitPyContainerTypeConverter {
  using value_type = typename T::value_type;

  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() ||
          EmitPyTypeConverter<value_type>::convert(arrayAttr[0])) {
        return convert(arrayAttr);
      }
      return {};
    }

    if constexpr (std::is_integral_v<value_type>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseBoolArrayAttr, mlir::DenseI8ArrayAttr,
                mlir::DenseI16ArrayAttr, mlir::DenseI32ArrayAttr,
                mlir::DenseI64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseIntElementsAttr>(
              [](mlir::DenseIntElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    if constexpr (std::is_floating_point_v<value_type>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseF32ArrayAttr, mlir::DenseF64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseFPElementsAttr>(
              [](mlir::DenseFPElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    return {};
  }

  static std::string convert(mlir::ArrayAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(*EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::enable_if_t<
      std::is_constructible_v<mlir::detail::DenseArrayAttrImpl<U>>, std::string>
  convert(mlir::detail::DenseArrayAttrImpl<U> attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr.asArrayRef()) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseFPElementsAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::string convert(llvm::ArrayRef<U> attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitPyTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

private:
  static std::string convert(const llvm::SmallVector<std::string> &values) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<T> << "[";
    llvm::interleaveComma(values, rso);
    rso << "]";
    return buf;
  }
};

template <typename T>
struct EmitPyTypeConverter<std::vector<T>>
    : public EmitPyContainerTypeConverter<std::vector<T>> {};

template <typename T>
struct EmitPyTypeConverter<::ttsl::SmallVector<T>>
    : public EmitPyContainerTypeConverter<::ttsl::SmallVector<T>> {};

template <typename T>
struct EmitPyTypeConverter<std::set<T>>
    : public EmitPyContainerTypeConverter<std::set<T>> {};

template <typename T, size_t k>
struct EmitPyTypeConverter<std::array<T, k>> {

  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() || EmitPyTypeConverter<T>::convert(arrayAttr[0])) {
        return convert(arrayAttr);
      }
      return {};
    }

    if constexpr (std::is_integral_v<T>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseBoolArrayAttr, mlir::DenseI8ArrayAttr,
                mlir::DenseI16ArrayAttr, mlir::DenseI32ArrayAttr,
                mlir::DenseI64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseIntElementsAttr>(
              [](mlir::DenseIntElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    if constexpr (std::is_floating_point_v<T>) {
      return llvm::TypeSwitch<mlir::Attribute, std::optional<std::string>>(attr)
          .Case<mlir::DenseF32ArrayAttr, mlir::DenseF64ArrayAttr>(
              [](auto denseArrayAttr) { return convert(denseArrayAttr); })
          .template Case<mlir::DenseFPElementsAttr>(
              [](mlir::DenseFPElementsAttr denseElementsAttr) {
                return convert(denseElementsAttr);
              })
          .Default([](auto) { return std::optional<std::string>{}; });
    }

    return {};
  }

  static std::optional<std::string> convert(mlir::ArrayAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (size_t i = 0; i < attr.size(); ++i) {
      auto element = EmitPyTypeConverter<T>::convert(attr[i]);
      if (!element) {
        return {};
      }
      result[i] = *element;
    }
    return convert(result);
  }

  template <typename U>
  static std::enable_if_t<
      std::is_constructible_v<mlir::detail::DenseArrayAttrImpl<U>>,
      std::optional<std::string>>
  convert(mlir::detail::DenseArrayAttrImpl<U> attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(attr[i]);
    }
    return convert(result);
  }

  static std::optional<std::string> convert(mlir::DenseIntElementsAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  static std::optional<std::string> convert(mlir::DenseFPElementsAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  template <typename U>
  static std::optional<std::string> convert(llvm::ArrayRef<U> attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (size_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitPyTypeConverter<T>::convert(attr[i]);
    }
    return convert(result);
  }

private:
  static std::string convert(const std::array<std::string, k> &values) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<std::array<T, k>> << "[";
    llvm::interleaveComma(values, rso);
    rso << "]";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::CoreRangeSet> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto coreRangeSetAttr =
            mlir::dyn_cast_if_present<ttnn::CoreRangeSetAttr>(attr)) {
      return convert(coreRangeSetAttr);
    }
    return {};
  }

  static std::string convert(ttnn::CoreRangeSetAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<::ttnn::CoreRangeSet>;
    rso << "[";
    rso << EmitPyTypeConverter<std::set<::ttnn::CoreRange>>::convert(
        attr.getCoreRanges());
    rso << "]";
    return buf;
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::ShardSpec> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardSpecAttr =
            mlir::dyn_cast_if_present<ttnn::ShardSpecAttr>(attr)) {
      return convert(shardSpecAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardSpecAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::ShardSpec>;
    rso << "(";
    rso << EmitPyTypeConverter<::ttnn::CoreRangeSet>::convert(
               attr.getCoreRangeSet())
        << ", ";
    rso << EmitPyTypeConverter<std::array<uint32_t, 2>>::convert(
               attr.getShape().getShape())
        << ", ";
    rso << EmitPyTypeConverter<::ttnn::types::ShardOrientation>::convert(
        attr.getShardOrientation());
    // ShardMode is modeled as optional in TTNN dialect, but it's either
    // required or defaulted to `Physical` in TTNN library.
    if (attr.getShardMode()) {
      rso << ", "
          << EmitPyTypeConverter<::ttnn::types::ShardMode>::convert(
                 attr.getShardMode());
    }
    rso << ")";

    return buf;
  }
};

template <typename T>
struct EmitPyTypeConverter<std::optional<T>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }
    return EmitPyTypeConverter<T>::convert(attr);
  }

  template <typename U>
  static std::string convert(std::optional<U> &&attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    return EmitPyTypeConverter<T>::convert(*attr);
  }

  template <typename U>
  static std::string convert(U &&attr) {
    return EmitPyTypeConverter<T>::convert(std::forward<U>(attr));
  }
};

template <>
struct EmitPyTypeConverter<::ttnn::MemoryConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto memoryConfigAttr =
            mlir::dyn_cast_if_present<ttnn::MemoryConfigAttr>(attr)) {
      return convert(memoryConfigAttr);
    }
    return {};
  }

  static std::string convert(ttnn::MemoryConfigAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<::ttnn::MemoryConfig> << "(";
    rso << EmitPyTypeConverter<::ttnn::TensorMemoryLayout>::convert(
               attr.getTensorMemoryLayout())
        << ", ";
    rso << EmitPyTypeConverter<::ttnn::BufferType>::convert(
               attr.getBufferType())
        << ", ";
    rso << EmitPyTypeConverter<std::optional<::ttnn::ShardSpec>>::convert(
        attr.getShardSpec());
    rso << ")";
    return buf;
  }
};

// This template struct retrieves the most relevant C++ type with a one-to-one
// Python type mapping for a given template type.
template <typename T>
struct TTNNTarget {
  using type = T;
};

template <typename T>
using TTNNTargetT = typename TTNNTarget<T>::type;

template <>
struct TTNNTarget<llvm::StringRef> {
  using type = std::string;
};

template <>
struct TTNNTarget<tt::ttnn::ShapeAttr> {
  using type = ::ttnn::Shape;
};

template <>
struct TTNNTarget<tt::ttcore::DataType> {
  using type = ::ttnn::DataType;
};

template <>
struct TTNNTarget<tt::ttcore::DataTypeAttr> {
  using type = ::ttnn::DataType;
};

template <>
struct TTNNTarget<tt::ttnn::BufferType> {
  using type = ::ttnn::BufferType;
};

template <>
struct TTNNTarget<tt::ttnn::BufferTypeAttr> {
  using type = ::ttnn::BufferType;
};

template <>
struct TTNNTarget<tt::ttnn::Layout> {
  using type = ::ttnn::Layout;
};

template <>
struct TTNNTarget<tt::ttnn::LayoutAttr> {
  using type = ::ttnn::Layout;
};

template <>
struct TTNNTarget<tt::ttnn::ShardSpecAttr> {
  using type = ::ttnn::ShardSpec;
};

template <>
struct TTNNTarget<tt::ttnn::TensorMemoryLayout> {
  using type = ::ttnn::TensorMemoryLayout;
};

template <>
struct TTNNTarget<tt::ttnn::TensorMemoryLayoutAttr> {
  using type = ::ttnn::TensorMemoryLayout;
};

template <>
struct TTNNTarget<tt::ttnn::MemoryConfigAttr> {
  using type = ::ttnn::MemoryConfig;
};

template <typename T>
struct IsMLIRType {
  static constexpr bool value = std::is_convertible_v<T, mlir::Attribute> ||
                                std::is_convertible_v<T, mlir::Value>;
};

template <typename T>
static constexpr bool IsMLIRTypeV = IsMLIRType<T>::value;

template <typename TTNNOp>
class EmitPyTTNNEmitter {
public:
  using OpAdaptor = typename TTNNOp::Adaptor;

  EmitPyTTNNEmitter(TTNNOp op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter)
      : op{op}, adaptor{adaptor}, rewriter{rewriter} {}

  EmitPyTTNNEmitter(const EmitPyTTNNEmitter &) = delete;
  EmitPyTTNNEmitter &operator=(const EmitPyTTNNEmitter &) = delete;

  template <typename TargetTy = void, typename SourceTy>
  mlir::Attribute emit(std::optional<SourceTy> attr,
                       std::string attrName = "") {
    if (!attr) {
      return emit(std::nullopt, attrName);
    }

    if constexpr (std::is_void_v<TargetTy>) {
      return emit(*attr, attrName);
    } else {
      return emit<TargetTy>(*attr, attrName);
    }
  }

  mlir::Attribute emit(std::nullopt_t, std::string attrName = "") {
    std::string keywordArg = attrName.empty() ? attrName : (attrName + "=");
    return rewriter.getType<emitpy::OpaqueAttr>(keywordArg +
                                                TypeNameV<std::nullopt_t>);
  }

  mlir::Attribute emit(mlir::Value val, std::string attrName = "") {
    if (!val) {
      return emit(std::nullopt, attrName);
    }

    unsigned index = getOperandIndex(val);
    operands.push_back(adaptor.getOperands()[index]);
    return rewriter.getIndexAttr(index);
  }

  mlir::Attribute emit(mlir::Operation::operand_range operands) {
    for (mlir::OpOperand &opOperand : op->getOpOperands()) {
      auto begin =
          std::next(op->getOperands().begin(), opOperand.getOperandNumber());
      if (mlir::Operation::operand_range(
              begin, std::next(begin, operands.size())) != operands) {
        continue;
      }
      unsigned index = opOperand.getOperandNumber();
      llvm::SmallVector<mlir::Value> values(
          adaptor.getOperands().begin() + index,
          adaptor.getOperands().begin() + index + operands.size());
      this->operands.push_back(createVector(values));
      return rewriter.getIndexAttr(index);
    }
    llvm_unreachable("Invalid operand range");
  }

  // Handles the case when source type is convertible to `mlir::Attribute` and
  // there exists a `TTNNTypeConverter` specialization for the TTNN target type
  // of the attribute.
  template <
      typename MLIRAttrTy, typename = std::void_t<TTNNTargetT<MLIRAttrTy>>,
      typename =
          std::enable_if_t<std::is_convertible_v<MLIRAttrTy, mlir::Attribute>>>
  mlir::Attribute emit(MLIRAttrTy attr, std::string attrName = "") {
    auto convertedValue =
        EmitPyTypeConverter<TTNNTargetT<MLIRAttrTy>>::convert(attr);

    std::string keywordArg = attrName.empty() ? attrName : (attrName + "=");
    if constexpr (std::is_same_v<decltype(convertedValue), std::string>) {
      return rewriter.getType<emitpy::OpaqueAttr>(keywordArg + convertedValue);
    } else if (convertedValue) {
      return rewriter.getType<emitpy::OpaqueAttr>(keywordArg + *convertedValue);
    }
    // It's assumed that the conversion might fail, in which case the result
    // will be `emitpy::OpaqueAttr("::std::nullopt")`.
    return emit(std::nullopt, attrName);
  }

  // Handles the case when source type is a non mlir::Attribute convertible
  // type.
  template <typename SourceTy>
  std::enable_if_t<!IsMLIRTypeV<SourceTy>, mlir::Attribute>
  emit(SourceTy attr, std::string attrName = "") {
    using TargetTy = TTNNTargetT<llvm::remove_cvref_t<SourceTy>>;
    auto result = EmitPyTypeConverter<TargetTy>::convert(attr);
    // It's assumed that the conversion will always succeed, if the result is
    // `std::optional<std::string>` we assume that it contains the converted
    // value.
    std::string keywordArg = attrName.empty() ? attrName : (attrName + "=");
    if constexpr (std::is_same_v<decltype(result),
                                 std::optional<std::string>>) {
      return rewriter.getType<emitpy::OpaqueAttr>(keywordArg + *result);
    }
    return rewriter.getType<emitpy::OpaqueAttr>(keywordArg + result);
  }

  mlir::Attribute getMemoryConfig(mlir::Value val) {
    auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(val.getType()).getEncoding());

    ttnn::BufferTypeAttr bufferTypeAttr = ttnn::BufferTypeAttr::get(
        layoutAttr.getContext(), layoutAttr.getBufferType());
    ttnn::TensorMemoryLayoutAttr tensorMemoryLayout = layoutAttr.getMemLayout();

    ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);

    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        layoutAttr.getContext(), tensorMemoryLayout, bufferTypeAttr,
        ttnn::utils::createShardSpecIfNeeded(layoutAttr,
                                             deviceAttr.getWorkerGrid()));

    return emit(memoryConfigAttr, "memory_config");
  }

  template <typename OpConversionPatternTy>
  mlir::Value replaceOp(OpConversionPatternTy &&opConversionPattern,
                        llvm::ArrayRef<mlir::Attribute> args) {
    auto resultTypes = llvm::to_vector(
        llvm::map_range(op->getResultTypes(), [&](Type type) -> Type {
          return opConversionPattern.getTypeConverter()->convertType(type);
        }));

    auto opName = op.getOperationName();
    if (opName == "ttnn.get_device") {
      opName = "my_get_device.DeviceGetter.get_device";
    }

    auto callOpaqueOp = rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultTypes, opName, operands, rewriter.getArrayAttr(args));

    assert(callOpaqueOp.getNumResults() <= 1 && "expected at most one result");
    if (callOpaqueOp.getNumResults() == 0) {
      return {};
    }
    return callOpaqueOp.getResult(0);
  }

private:
  mlir::Value createVector(ValueRange operands) {
    return rewriter
        .create<emitpy::CallOpaqueOp>(
            op.getLoc(),
            emitpy::OpaqueType::get(rewriter.getContext(),
                                    TypeNameV<std::vector<::ttnn::Tensor>>),
            "util_create_list", nullptr, nullptr, operands)
        ->getResult(0);
  }

  unsigned getOperandIndex(mlir::Value value) {
    mlir::OpOperand *opOperand = std::find_if(
        op->getOpOperands().begin(), op->getOpOperands().end(),
        [&](OpOperand &operand) { return operand.get() == value; });

    return opOperand->getOperandNumber();
  }

  TTNNOp op;
  OpAdaptor adaptor;
  ConversionPatternRewriter &rewriter;
  llvm::SmallVector<mlir::Value> operands;
};

// Helper function that serves as an alternative to the
// `emit<std::variant<...>>` member function of the `EmitPyTTNNEmitter` class.
// For example, instead of calling `emit<std::variant<int32_t, float>>(attr)`,
// one can call `emit<int32_t>(attr) | emit<float>(attr)`.
inline mlir::Attribute operator|(mlir::Attribute lhs, mlir::Attribute rhs) {
  const mlir::Attribute nulloptAttr = tt::emitpy::OpaqueAttr::get(
      lhs.getContext(), tt::ttnn_to_emitpy::TypeNameV<std::nullopt_t>);
  if (!lhs || lhs == nulloptAttr) {
    return rhs;
  }
  return lhs;
}

} // namespace ttnn_to_emitpy
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITPY_EMITPYCONVERSION_H
