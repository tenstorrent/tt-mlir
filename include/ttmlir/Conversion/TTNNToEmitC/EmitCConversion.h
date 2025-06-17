// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H

#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

// This namespace contains mock definitions of TTNN types for the purpose of
// conversion.
namespace ttnn {
template <typename T>
struct SmallVector {
  using value_type = T;
};

struct Shape;

struct ShardSpec;
struct CoreRangeSet;
struct CoreRange;
struct CoreCoord;

struct DataType;
struct TensorMemoryLayout;
struct Layout;

namespace types {
struct ShardOrientation;
struct ShardMode;
} // namespace types

namespace distributed {
struct MeshDevice;
} // namespace distributed
struct IDevice;

struct Tensor;

namespace operations {
namespace unary {

// Mock definition of VecMode enum from tt-metal
enum class VecMode {
  None = 0,
  R = 1,
  C = 2,
  RC = 4,
  RC_custom = 6,
  Invalid = 0xFF,
};

} // namespace unary

namespace creation::detail {
using OptionalMeshDevice =
    std::optional<std::reference_wrapper<distributed::MeshDevice>>;
} // namespace creation::detail

namespace conv::conv2d {
struct Conv2dConfig;
} // namespace conv::conv2d
} // namespace operations
} // namespace ttnn

namespace mlir {
namespace tt {
namespace ttnn_to_emitc {

template <typename T>
struct TypeName;

template <typename... Types>
struct JoinTypeNames;

template <typename T>
const std::string TypeNameV = TypeName<T>::value;

template <typename... Types>
const std::string JoinTypeNamesV = JoinTypeNames<Types...>::value;

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

template <>
struct TypeName<std::nullopt_t> {
  // This is a special case, as std::nullopt is not a type, but is the only
  // value of type std::nullopt_t.
  inline static const std::string value = "::std::nullopt";
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

template <typename T>
struct TypeName<std::set<T>> {
  inline static const std::string value = "::std::set<" + TypeNameV<T> + ">";
};

template <typename T>
struct TypeName<std::optional<T>> {
  inline static const std::string value =
      "::std::optional<" + TypeNameV<T> + ">";
};

template <typename T>
struct TypeName<std::reference_wrapper<T>> {
  inline static const std::string value =
      "::std::reference_wrapper<" + TypeNameV<T> + ">";
};

template <typename... Types>
struct TypeName<std::tuple<Types...>> {
  inline static const std::string value =
      "::std::tuple<" + JoinTypeNamesV<Types...> + ">";
};

template <typename... Types>
struct TypeName<std::variant<Types...>> {
  inline static const std::string value =
      "::std::variant<" + JoinTypeNamesV<Types...> + ">";
};

template <>
struct TypeName<::ttnn::distributed::MeshDevice> {
  inline static const std::string value = "::ttnn::distributed::MeshDevice";
};

template <>
struct TypeName<::ttnn::IDevice> {
  inline static const std::string value = "::ttnn::IDevice";
};

template <>
struct TypeName<::ttnn::operations::creation::detail::OptionalMeshDevice> {
  // Following results in empty string, so hardcoded value is used instead
  // TypeNameV<std::optional<std::reference_wrapper<::ttnn::distributed::MeshDevice>>>
  inline static const std::string value =
      "::std::optional<::std::reference_wrapper<::ttnn::distributed::"
      "MeshDevice>>";
};

template <>
struct TypeName<::ttnn::Tensor> {
  inline static const std::string value = "::ttnn::Tensor";
};

template <>
struct TypeName<::ttnn::operations::conv::conv2d::Conv2dConfig> {
  inline static const std::string value =
      "::ttnn::operations::conv::conv2d::Conv2dConfig";
};

template <>
struct TypeName<::ttnn::CoreCoord> {
  inline static const std::string value = "::ttnn::CoreCoord";
};

template <>
struct TypeName<::ttnn::CoreRange> {
  inline static const std::string value = "::ttnn::CoreRange";
};

template <>
struct TypeName<::ttnn::CoreRangeSet> {
  inline static const std::string value = "::ttnn::CoreRangeSet";
};

template <>
struct TypeName<::ttnn::ShardSpec> {
  inline static const std::string value = "::ttnn::ShardSpec";
};

template <>
struct TypeName<::ttnn::types::ShardOrientation> {
  inline static const std::string value = "::ttnn::types::ShardOrientation";
};

template <>
struct TypeName<::ttnn::types::ShardMode> {
  inline static const std::string value = "::ttnn::types::ShardMode";
};

template <>
struct TypeName<::ttnn::DataType> {
  inline static const std::string value = "::ttnn::DataType";
};

template <>
struct TypeName<::ttnn::TensorMemoryLayout> {
  inline static const std::string value = "::ttnn::TensorMemoryLayout";
};

template <>
struct TypeName<::ttnn::Layout> {
  inline static const std::string value = "::ttnn::Layout";
};

template <>
struct JoinTypeNames<> {
  inline static const std::string value = "";
};

template <typename T>
struct JoinTypeNames<T> {
  inline static const std::string value = TypeNameV<T>;
};

template <typename T, typename... Rest>
struct JoinTypeNames<T, Rest...> {
  inline static const std::string value =
      TypeNameV<T> + ", " + JoinTypeNamesV<Rest...>;
};

template <typename T, typename Enable = void>
struct EmitCTypeConverter;

template <>
struct EmitCTypeConverter<bool> {
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

  static std::string convert(bool value) { return value ? "true" : "false"; }
};

template <>
struct EmitCTypeConverter<std::string> {
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

  template <typename U>
  static std::enable_if_t<std::is_integral_v<U>, std::string> convert(U value) {
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

  template <typename U>
  static std::enable_if_t<std::is_floating_point_v<U>, std::string>
  convert(U value) {
    if (std::isfinite(value)) {
      std::string result = std::to_string(static_cast<T>(value));
      if constexpr (std::is_same_v<T, float>) {
        result.append("f");
      }
      return result;
    }

    if (std::isinf(value)) {
      std::string result = value > 0 ? "" : "-";
      result.append("::std::numeric_limits<" + TypeNameV<T> + ">::infinity()");
      return result;
    }

    if (std::isnan(value)) {
      return "::std::numeric_limits<" + TypeNameV<T> + ">::quiet_NaN()";
    }

    llvm_unreachable("Unknown class of floating point value");
  }
};

template <>
struct EmitCTypeConverter<::ttnn::CoreCoord> {
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
    rso << "{";
    rso << EmitCTypeConverter<size_t>::convert(attr.getX()) << ", ";
    rso << EmitCTypeConverter<size_t>::convert(attr.getY());
    rso << "}";

    return buf;
  }
};

template <>
struct EmitCTypeConverter<::ttnn::CoreRange> {
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
    rso << "{";
    rso << EmitCTypeConverter<::ttnn::CoreCoord>::convert(attr.getStartCoord())
        << ", ";
    rso << EmitCTypeConverter<::ttnn::CoreCoord>::convert(attr.getEndCoord());
    rso << "}";

    return buf;
  }
};

template <>
struct EmitCTypeConverter<::ttnn::types::ShardOrientation> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardOrientationAttr =
            mlir::dyn_cast_if_present<ttnn::ShardOrientationAttr>(attr)) {
      return convert(shardOrientationAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardOrientationAttr attr) {
    assert(attr &&
           "expected non-null attribute, call "
           "EmitCTypeConverter<std::optional<::ttnn::types::ShardOrientation>>:"
           ":convert(attr) if attribute is optional");
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::ShardOrientation attr) {
    switch (attr) {
    case ttnn::ShardOrientation::RowMajor:
      return TypeNameV<::ttnn::types::ShardOrientation> + "::ROW_MAJOR";
    case ttnn::ShardOrientation::ColMajor:
      return TypeNameV<::ttnn::types::ShardOrientation> + "::COL_MAJOR";
    }

    llvm_unreachable("Unknown ttnn::ShardOrientation");
  }
};

template <>
struct EmitCTypeConverter<::ttnn::types::ShardMode> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto shardModeAttr =
            mlir::dyn_cast_if_present<ttnn::ShardModeAttr>(attr)) {
      return convert(shardModeAttr);
    }
    return {};
  }

  static std::string convert(ttnn::ShardModeAttr attr) {
    assert(attr && "expected non-null attribute, call "
                   "EmitCTypeConverter<std::optional<::ttnn::types::ShardMode>>"
                   "::convert(attr) if attribute is optional");
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::ShardMode attr) {
    switch (attr) {
    case ttnn::ShardMode::Physical:
      return TypeNameV<::ttnn::types::ShardMode> + "::PHYSICAL";
    case ttnn::ShardMode::Logical:
      return TypeNameV<::ttnn::types::ShardMode> + "::LOGICAL";
    }

    llvm_unreachable("Unknown ttnn::ShardMode");
  }
};

template <>
struct EmitCTypeConverter<::ttnn::DataType> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto dataTypeAttr = mlir::dyn_cast_if_present<tt::DataTypeAttr>(attr)) {
      return convert(dataTypeAttr);
    }
    return {};
  }

  static std::string convert(tt::DataTypeAttr attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }

    return convert(attr.getValue());
  }

  static std::string convert(tt::DataType attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::DataType> << "::";
    switch (attr) {
    case tt::DataType::BFloat16:
      rso << "BFLOAT16";
      break;
    case tt::DataType::Float32:
      rso << "FLOAT32";
      break;
    case tt::DataType::UInt32:
      rso << "UINT32";
      break;
    case tt::DataType::BFP_BFloat8:
      rso << "BFLOAT8_B";
      break;
    case tt::DataType::BFP_BFloat4:
      rso << "BFLOAT4_B";
      break;
    case tt::DataType::UInt8:
      rso << "UINT8";
      break;
    case tt::DataType::UInt16:
      rso << "UINT16";
      break;
    case tt::DataType::Int32:
      rso << "INT32";
      break;
    case tt::DataType::Float16:
    case tt::DataType::BFP_Float2:
    case tt::DataType::BFP_Float4:
    case tt::DataType::BFP_Float8:
    case tt::DataType::BFP_BFloat2:
      llvm_unreachable("Unsupported ttnn::DataType");
    }

    return buf;
  }
};

template <>
struct EmitCTypeConverter<::ttnn::TensorMemoryLayout> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto tensorMemoryLayoutAttr =
            mlir::dyn_cast_if_present<ttnn::TensorMemoryLayoutAttr>(attr)) {
      return convert(tensorMemoryLayoutAttr);
    }
    return {};
  }

  static std::string convert(ttnn::TensorMemoryLayoutAttr attr) {
    // TODO (azecevic): There is a dissonance between the way we model
    // TensorMemoryLayout in TTNN dialect and TTNN library. This should be fixed
    // with https://github.com/tenstorrent/tt-mlir/issues/2521. For now, we
    // default to Interleaved, which is default value in TTNN library.
    if (!attr) {
      return convert(ttnn::TensorMemoryLayout::Interleaved);
    }
    return convert(attr.getValue());
  }

  static std::string convert(ttnn::TensorMemoryLayout attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    rso << TypeNameV<::ttnn::TensorMemoryLayout> << "::";
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
    case ttnn::TensorMemoryLayout::SingleBank:
      rso << "SINGLE_BANK";
      break;
    case ttnn::TensorMemoryLayout::WidthSharded:
      rso << "WIDTH_SHARDED";
      break;
    }

    return buf;
  }
};

template <>
struct EmitCTypeConverter<::ttnn::Layout> {
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

    rso << TypeNameV<::ttnn::Layout> << "::";
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

// Convert container types (std::vector, ttnn::SmallVector, etc.).
template <typename T>
struct EmitCContainerTypeConverter {
  using value_type = typename T::value_type;

  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() ||
          EmitCTypeConverter<value_type>::convert(arrayAttr[0])) {
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
      result.push_back(*EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::enable_if_t<
      std::is_constructible_v<mlir::detail::DenseArrayAttrImpl<U>>, std::string>
  convert(mlir::detail::DenseArrayAttrImpl<U> attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr.asArrayRef()) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  static std::string convert(mlir::DenseFPElementsAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

  template <typename U>
  static std::string convert(llvm::ArrayRef<U> attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return convert(result);
  }

private:
  static std::string convert(const llvm::SmallVector<std::string> &values) {
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

template <typename T>
struct EmitCTypeConverter<std::set<T>>
    : public EmitCContainerTypeConverter<std::set<T>> {};

template <typename T, size_t k>
struct EmitCTypeConverter<std::array<T, k>> {

  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      if (arrayAttr.empty() || EmitCTypeConverter<T>::convert(arrayAttr[0])) {
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
      auto element = EmitCTypeConverter<T>::convert(attr[i]);
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
      result[i] = EmitCTypeConverter<T>::convert(attr[i]);
    }
    return convert(result);
  }

  static std::optional<std::string> convert(mlir::DenseIntElementsAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return convert(result);
  }

  static std::optional<std::string> convert(mlir::DenseFPElementsAttr attr) {
    if (attr.size() != k) {
      return {};
    }

    std::array<std::string, k> result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
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
      result[i] = EmitCTypeConverter<T>::convert(attr[i]);
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
    auto tryFirst = EmitCTypeConverter<First>::convert(attr);
    if constexpr (std::is_same_v<decltype(tryFirst), std::string>) {
      return tryFirst;
    } else {
      static_assert(
          std::is_same_v<decltype(tryFirst), std::optional<std::string>>);
      if (tryFirst) {
        return *tryFirst;
      }
    }

    if constexpr (sizeof...(Rest) > 0) {
      return EmitCTypeConverter<std::variant<Rest...>>::convert(attr);
    }

    llvm_unreachable("Invalid variant type");
  }
};

template <>
struct EmitCTypeConverter<::ttnn::CoreRangeSet> {
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
    rso << "{";
    rso << EmitCTypeConverter<std::set<::ttnn::CoreRange>>::convert(
        attr.getCoreRanges());
    rso << "}";
    return buf;
  }
};

template <>
struct EmitCTypeConverter<::ttnn::ShardSpec> {
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
    rso << "{";
    rso << EmitCTypeConverter<::ttnn::CoreRangeSet>::convert(
               attr.getCoreRangeSet())
        << ", ";
    rso << EmitCTypeConverter<std::array<uint32_t, 2>>::convert(
               attr.getShape().getShape())
        << ", ";
    rso << EmitCTypeConverter<::ttnn::types::ShardOrientation>::convert(
        attr.getShardOrientation());
    // ShardMode is modeled as optional in TTNN dialect, but it's either
    // required or defaulted to `Physical` in TTNN library.
    if (attr.getShardMode()) {
      rso << ", "
          << EmitCTypeConverter<::ttnn::types::ShardMode>::convert(
                 attr.getShardMode());
    }
    rso << "}";

    return buf;
  }
};

template <typename T>
struct EmitCTypeConverter<std::optional<T>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }
    return EmitCTypeConverter<T>::convert(attr);
  }

  template <typename U>
  static std::string convert(std::optional<U> &&attr) {
    if (!attr) {
      return TypeNameV<std::nullopt_t>;
    }
    return EmitCTypeConverter<T>::convert(*attr);
  }

  template <typename U>
  static std::string convert(U &&attr) {
    return EmitCTypeConverter<T>::convert(std::forward<U>(attr));
  }
};

template <>
struct EmitCTypeConverter<::ttnn::operations::conv::conv2d::Conv2dConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (auto conv2dConfigAttr =
            mlir::dyn_cast_if_present<ttnn::Conv2dConfigAttr>(attr)) {
      return convert(conv2dConfigAttr);
    }
    return {};
  }

  static std::string convert(ttnn::Conv2dConfigAttr attr) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);

    bool firstElement = true;
    rso << TypeNameV<::ttnn::operations::conv::conv2d::Conv2dConfig> << "{";
    if (attr.getDtype()) {
      rso << (firstElement ? "" : ", ") << ".dtype = "
          << EmitCTypeConverter<::ttnn::DataType>::convert(*attr.getDtype());
      firstElement = false;
    }
    if (attr.getWeightsDtype()) {
      rso << (firstElement ? "" : ", ") << ".weights_dtype = "
          << EmitCTypeConverter<::ttnn::DataType>::convert(
                 *attr.getWeightsDtype());
      firstElement = false;
    }
    if (attr.getActivation()) {
      rso << (firstElement ? "" : ", ") << ".activation = "
          << EmitCTypeConverter<std::string>::convert(attr.getActivation());
      firstElement = false;
    }
    if (attr.getDeallocateActivation()) {
      rso << (firstElement ? "" : ", ") << ".deallocate_activation = "
          << EmitCTypeConverter<bool>::convert(attr.getDeallocateActivation());
      firstElement = false;
    }
    if (attr.getReallocateHaloOutput()) {
      rso << (firstElement ? "" : ", ") << ".reallocate_halo_output = "
          << EmitCTypeConverter<bool>::convert(attr.getReallocateHaloOutput());
      firstElement = false;
    }
    if (attr.getActBlockHOverride()) {
      rso << (firstElement ? "" : ", ") << ".act_block_h_override = "
          << EmitCTypeConverter<uint32_t>::convert(
                 *attr.getActBlockHOverride());
      firstElement = false;
    }
    if (attr.getActBlockWDiv()) {
      rso << (firstElement ? "" : ", ") << ".act_block_w_div = "
          << EmitCTypeConverter<uint32_t>::convert(*attr.getActBlockWDiv());
      firstElement = false;
    }
    if (attr.getReshardIfNotOptimal()) {
      rso << (firstElement ? "" : ", ") << ".reshard_if_not_optimal = "
          << EmitCTypeConverter<bool>::convert(attr.getReshardIfNotOptimal());
      firstElement = false;
    }
    if (attr.getOverrideShardingConfig()) {
      rso << (firstElement ? "" : ", ") << ".override_sharding_config = "
          << EmitCTypeConverter<bool>::convert(
                 attr.getOverrideShardingConfig());
      firstElement = false;
    }
    if (attr.getShardLayout()) {
      rso << (firstElement ? "" : ", ") << ".shard_layout = "
          << EmitCTypeConverter<::ttnn::TensorMemoryLayout>::convert(
                 *attr.getShardLayout());
      firstElement = false;
    }
    if (attr.getCoreGrid()) {
      rso << (firstElement ? "" : ", ") << ".core_grid = "
          << EmitCTypeConverter<::ttnn::CoreRangeSet>::convert(
                 attr.getCoreGrid());
      firstElement = false;
    }
    if (attr.getTransposeShards()) {
      rso << (firstElement ? "" : ", ") << ".transpose_shards = "
          << EmitCTypeConverter<bool>::convert(attr.getTransposeShards());
      firstElement = false;
    }
    if (attr.getOutputLayout()) {
      rso << (firstElement ? "" : ", ") << ".output_layout = "
          << EmitCTypeConverter<::ttnn::Layout>::convert(
                 *attr.getOutputLayout());
      firstElement = false;
    }
    if (attr.getPreprocessWeightsOnDevice()) {
      rso << (firstElement ? "" : ", ") << ".preprocess_weights_on_device = "
          << EmitCTypeConverter<bool>::convert(
                 attr.getPreprocessWeightsOnDevice());
    }
    rso << "}";
    return buf;
  }
};

// This template struct is used to retrieve the single most relevant C++ type in
// TTNN for a given template type.
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
struct TTNNTarget<llvm::APFloat> {
  using type = float;
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

inline std::string convert(tt::DataType attr) {
  // TODO (azecevic): Will be deprecated!
  // https://github.com/tenstorrent/tt-mlir/issues/3635
  return EmitCTypeConverter<::ttnn::DataType>::convert(attr);
}

inline std::string convert(tt::DataTypeAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
  }

  return convert(attr.getValue());
}

inline std::string convert(ttnn::Layout attr) {
  // TODO (azecevic): Will be deprecated!
  // https://github.com/tenstorrent/tt-mlir/issues/3635
  return EmitCTypeConverter<::ttnn::Layout>::convert(attr);
}

inline std::string convert(ttnn::LayoutAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
  }

  return convert(attr.getValue());
}

inline std::string convert(ttnn::TensorMemoryLayout attr) {
  // TODO (azecevic): Will be deprecated!
  // https://github.com/tenstorrent/tt-mlir/issues/3635
  return EmitCTypeConverter<::ttnn::TensorMemoryLayout>::convert(attr);
}

inline std::string convert(ttnn::TensorMemoryLayoutAttr attr) {
  // TODO (azecevic): There is a dissonance between the way we model
  // TensorMemoryLayout in TTNN dialect and TTNN library. This should be fixed
  // with https://github.com/tenstorrent/tt-mlir/issues/2521. For now, we
  // default to Interleaved, which is default value in TTNN library.
  if (!attr) {
    return convert(ttnn::TensorMemoryLayout::Interleaved);
  }

  return convert(attr.getValue());
}

inline std::string convert(ttnn::BufferType attr) {
  switch (attr) {
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

inline std::string convert(ttnn::BufferTypeAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
  }

  return convert(attr.getValue());
}

inline std::string convert(ttnn::MemoryConfigAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
  }

  std::string buf;
  llvm::raw_string_ostream rso(buf);
  rso << "::ttnn::MemoryConfig{";
  rso << convert(attr.getTensorMemoryLayout()) << ", ";
  rso << convert(attr.getBufferType()) << ", ";
  rso << EmitCTypeConverter<std::optional<::ttnn::ShardSpec>>::convert(
      attr.getShardSpec());
  rso << "}";
  return buf;
}

template <typename T>
struct IsMLIRType {
  static constexpr bool value = std::is_convertible_v<T, mlir::Attribute> ||
                                std::is_convertible_v<T, mlir::Value>;
};

template <typename T>
static constexpr bool IsMLIRTypeV = IsMLIRType<T>::value;

template <typename TTNNOp>
class EmitCTTNNEmitter {
public:
  using OpAdaptor = typename TTNNOp::Adaptor;

  EmitCTTNNEmitter(TTNNOp op, OpAdaptor adaptor,
                   mlir::ConversionPatternRewriter &rewriter)
      : op{op}, adaptor{adaptor}, rewriter{rewriter} {}

  EmitCTTNNEmitter(const EmitCTTNNEmitter &) = delete;
  EmitCTTNNEmitter &operator=(const EmitCTTNNEmitter &) = delete;

  mlir::Attribute emit(tt::ttnn::ShapeAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(convert(attr));
  }

  mlir::Attribute emit(tt::DataType attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(convert(attr));
  }

  mlir::Attribute emit(tt::DataTypeAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(
        tt::ttnn_to_emitc::convert(attr));
  }

  mlir::Attribute emit(tt::ttnn::Layout attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(convert(attr));
  }

  mlir::Attribute emit(tt::ttnn::LayoutAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(convert(attr));
  }

  mlir::Attribute emit(ttnn::TensorMemoryLayout attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(convert(attr));
  }

  mlir::Attribute emit(ttnn::TensorMemoryLayoutAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(convert(attr));
  }

  mlir::Attribute emit(tt::ttnn::MemoryConfigAttr attr) {
    return rewriter.getType<emitc::OpaqueAttr>(convert(attr));
  }

  template <typename TargetTy = void, typename SourceTy>
  mlir::Attribute emit(std::optional<SourceTy> attr) {
    if (!attr) {
      return emit(std::nullopt);
    }

    if constexpr (std::is_void_v<TargetTy>) {
      return emit(*attr);
    } else {
      return emit<TargetTy>(*attr);
    }
  }

  mlir::Attribute emit(std::nullopt_t) {
    return rewriter.getType<emitc::OpaqueAttr>(TypeNameV<std::nullopt_t>);
  }

  mlir::Attribute emit(mlir::Value val) {
    if (!val) {
      return emit(std::nullopt);
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

  template <typename TargetTy = void>
  mlir::Attribute emit(std::nullptr_t) {
    if constexpr (std::is_void_v<TargetTy>) {
      return rewriter.getType<emitc::OpaqueAttr>("nullptr");
    } else {
      return rewriter.getType<emitc::OpaqueAttr>(
          "static_cast<" + TypeNameV<TargetTy> + " *>(nullptr)");
    }
  }

  // Handles the case when source type is convertible to mlir::Attribute type
  // and source and target types are in many-to-many relationship (i.e.
  // {mlir::ArrayAttr, mlir::DenseI32ArrayAttr} to {std::vector<uint32_t>,
  // ttnn::SmallVector<int32_t>}).
  template <typename TargetTy>
  mlir::Attribute emit(mlir::Attribute attr) {
    auto convertedValue = EmitCTypeConverter<TargetTy>::convert(attr);
    if constexpr (std::is_same_v<decltype(convertedValue), std::string>) {
      return rewriter.getType<emitc::OpaqueAttr>(convertedValue);
    } else if (convertedValue) {
      return rewriter.getType<emitc::OpaqueAttr>(*convertedValue);
    }
    // It's assumed that the conversion might fail, in which case the result
    // will be `emitc::OpaqueAttr("::std::nullopt")`.
    return emit(std::nullopt);
  }

  // Handles the case when source type is a non mlir::Attribute convertible type
  // and source and target types are in many-to-many relationship (i.e.
  // llvm::ArrayRef<T> to std::vector<U>). For convenience, by default
  // `TargetTy` is the same as `SourceTy`, for cases where we already have an
  // appropriate C++ type.
  template <typename TargetTy = void, typename SourceTy>
  std::enable_if_t<!IsMLIRTypeV<SourceTy>, mlir::Attribute>
  emit(SourceTy &&attr) {
    using ActualTargetTy =
        std::conditional_t<std::is_void_v<TargetTy>,
                           TTNNTargetT<llvm::remove_cvref_t<SourceTy>>,
                           TargetTy>;
    auto result = EmitCTypeConverter<ActualTargetTy>::convert(
        std::forward<SourceTy>(attr));
    // It's assumed that the conversion will always succeed, if the result is
    // `std::optional<std::string>` we assume that it contains the converted
    // value.
    if constexpr (std::is_same_v<decltype(result),
                                 std::optional<std::string>>) {
      return rewriter.getType<emitc::OpaqueAttr>(*result);
    }
    return rewriter.getType<emitc::OpaqueAttr>(result);
  }

  // Handles conversion of DeviceType objects to:
  // - ::ttnn::distributed::MeshDevice *
  // - ::ttnn::distributed::MeshDevice
  // - ::ttnn::operations::creation::detail::OptionalMeshDevice
  //
  // Will return `std::nullopt` if DeviceType is null
  //
  template <typename TargetTy = ::ttnn::distributed::MeshDevice *>
  mlir::Attribute
  emit(::mlir::TypedValue<::mlir::tt::ttnn::DeviceType> device) {
    if (!device) {
      return emit(std::nullopt);
    }

    if constexpr (std::is_same_v<TargetTy, ::ttnn::distributed::MeshDevice *> ||
                  std::is_same_v<TargetTy, ::ttnn::distributed::MeshDevice>) {
      unsigned index = getOperandIndex(device);
      operands.push_back(adaptor.getOperands()[index]);

      return rewriter.getIndexAttr(index);
    } else if constexpr (std::is_same_v<TargetTy,
                                        ::ttnn::operations::creation::detail::
                                            OptionalMeshDevice>) {
      unsigned index = getOperandIndex(device);
      mlir::Value deviceValueFromOperandsList = adaptor.getOperands()[index];

      // optional<reference_wrapper<MeshDevice>> x = *device_ptr
      emitc::ApplyOp meshDeviceOp = rewriter.create<emitc::ApplyOp>(
          op.getLoc(),
          rewriter.getType<emitc::OpaqueType>(
              TypeNameV<
                  ::ttnn::operations::creation::detail::OptionalMeshDevice>),
          "*", // Dereference operator
          deviceValueFromOperandsList);

      operands.push_back(meshDeviceOp->getResult(0));
      return rewriter.getIndexAttr(operands.size() - 1);
    } else {
      llvm_unreachable("Unknown TargetTy");
    }
  }

  template <typename OpConversionPatternTy>
  emitc::CallOpaqueOp replaceOp(OpConversionPatternTy &&opConversionPattern,
                                llvm::ArrayRef<mlir::Attribute> args) {
    // Special handling for Conv2dOp and ConvTranspose2dOp. These ops have a
    // different return type than the other TTNN ops. They return
    // `std::tuple<::ttnn::Tensor, uint32_t, uint32_t, ::ttnn::Tensor,
    // std::optional<::ttnn::Tensor>>`, but we want to return only the first
    // element of the tuple.
    if constexpr (std::is_same_v<TTNNOp, tt::ttnn::Conv2dOp> ||
                  std::is_same_v<TTNNOp, tt::ttnn::ConvTranspose2dOp>) {
      using OutputHeight = std::uint32_t;
      using OutputWidth = std::uint32_t;
      using ReturnTy = std::variant<
          ::ttnn::Tensor,
          std::tuple<::ttnn::Tensor, std::tuple<OutputHeight, OutputWidth>>,
          std::tuple<::ttnn::Tensor,
                     std::tuple<::ttnn::Tensor, std::optional<::ttnn::Tensor>>>,
          std::tuple<
              ::ttnn::Tensor, std::tuple<OutputHeight, OutputWidth>,
              std::tuple<::ttnn::Tensor, std::optional<::ttnn::Tensor>>>>;

      auto opResult = rewriter.create<emitc::CallOpaqueOp>(
          op.getLoc(), rewriter.getType<emitc::OpaqueType>(TypeNameV<ReturnTy>),
          opConversionPattern.convertOpName(op), rewriter.getArrayAttr(args),
          nullptr, adaptor.getOperands());

      return rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
          op, rewriter.getType<emitc::OpaqueType>(TypeNameV<::ttnn::Tensor>),
          "::std::get", rewriter.getArrayAttr({rewriter.getIndexAttr(0)}),
          rewriter.getArrayAttr(
              {rewriter.getIntegerAttr(rewriter.getI32Type(), 0)}),
          opResult.getResult(0));
    }

    auto resultTypes = llvm::to_vector(
        llvm::map_range(op->getResultTypes(), [&](Type type) -> Type {
          return opConversionPattern.getTypeConverter()->convertType(type);
        }));
    return rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultTypes, opConversionPattern.convertOpName(op),
        rewriter.getArrayAttr(args), nullptr, operands);
  }

  // TODO (azecevic): This is a temporary solution for handling the case when
  // the value of the MemoryConfigAttr is nullptr. This should be removed once
  // https://github.com/tenstorrent/tt-mlir/issues/2415 lands.
  mlir::Attribute getMemoryConfig(mlir::Value val) {
    auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(val.getType()).getEncoding());

    ttnn::BufferTypeAttr bufferTypeAttr = ttnn::BufferTypeAttr::get(
        layoutAttr.getContext(), layoutAttr.getBufferType());
    ttnn::TensorMemoryLayoutAttr tensorMemoryLayout = layoutAttr.getMemLayout();

    DeviceAttr deviceAttr = lookupDevice(op);

    ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
        layoutAttr.getContext(), tensorMemoryLayout, bufferTypeAttr,
        ttnn::utils::createShardSpecIfNeeded(layoutAttr,
                                             deviceAttr.getWorkerGrid()));

    return emit(memoryConfigAttr);
  }

  // TODO (azecevic): This is a temporary solution for handling the case when
  // the value of the Conv2dConfigAttr is nullptr. This should be removed once
  // https://github.com/tenstorrent/tt-mlir/issues/2852 lands.
  mlir::Attribute getConv2dConfig(mlir::Value input, mlir::Value weight) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());

    auto inputLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
    auto weightLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(weightType.getEncoding());

    DataType inputDataType = inputLayoutAttr.getDataType();
    DataType weightDataType = weightLayoutAttr.getDataType();

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<::ttnn::operations::conv::conv2d::Conv2dConfig> << "{";
    rso << ".dtype = " << convert(inputDataType) << ", ";
    rso << ".weights_dtype = " << convert(weightDataType);
    rso << "}";

    return rewriter.getType<emitc::OpaqueAttr>(buf);
  }

private:
  mlir::Value createVector(ValueRange operands) {
    tt::ttnn_to_emitc::utils::insertVecCreateFnIfNotExists(rewriter, op);

    return rewriter
        .create<emitc::CallOpaqueOp>(
            op.getLoc(),
            emitc::OpaqueType::get(rewriter.getContext(),
                                   TypeNameV<std::vector<::ttnn::Tensor>>),
            tt::ttnn_to_emitc::utils::kCreateVectorFunctionName, nullptr,
            nullptr, operands)
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

} // namespace ttnn_to_emitc
} // namespace tt

// Helper function that serves as an alternative to the
// `emit<std::variant<...>>` member function of the `EmitCTTNNEmitter` class.
// For example, instead of calling `emit<std::variant<int32_t, float>>(attr)`,
// one can call `emit<int32_t>(attr) | emit<float>(attr)`.
inline mlir::Attribute operator|(mlir::Attribute lhs, mlir::Attribute rhs) {
  const mlir::Attribute nulloptAttr = emitc::OpaqueAttr::get(
      lhs.getContext(), tt::ttnn_to_emitc::TypeNameV<std::nullopt_t>);
  if (!lhs || lhs == nulloptAttr) {
    return rhs;
  }
  return lhs;
}

} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
