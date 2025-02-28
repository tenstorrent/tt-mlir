// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
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

  static std::string convert(mlir::BoolAttr attr) {
    return convert(attr.getValue());
  }

  static std::string convert(bool value) { return value ? "true" : "false"; }
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
    return std::to_string(static_cast<T>(value));
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
      if (auto denseBoolArrayAttr =
              mlir::dyn_cast<mlir::DenseBoolArrayAttr>(attr)) {
        return convert(denseBoolArrayAttr);
      }
      if (auto denseI8ArrayAttr =
              mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr)) {
        return convert(denseI8ArrayAttr);
      }
      if (auto denseI16ArrayAttr =
              mlir::dyn_cast<mlir::DenseI16ArrayAttr>(attr)) {
        return convert(denseI16ArrayAttr);
      }
      if (auto denseI32ArrayAttr =
              mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
        return convert(denseI32ArrayAttr);
      }
      if (auto denseI64ArrayAttr =
              mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
        return convert(denseI64ArrayAttr);
      }
      if (auto denseIntAttr =
              mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
        return convert(denseIntAttr);
      }
    }

    if constexpr (std::is_floating_point_v<value_type>) {
      if (auto denseF32ArrayAttr =
              mlir::dyn_cast<mlir::DenseF32ArrayAttr>(attr)) {
        return convert(denseF32ArrayAttr);
      }
      if (auto denseF64ArrayAttr =
              mlir::dyn_cast<mlir::DenseF64ArrayAttr>(attr)) {
        return convert(denseF64ArrayAttr);
      }
      if (auto denseFPAttr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(attr)) {
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

  template <typename U>
  static std::enable_if_t<
      std::is_constructible_v<mlir::detail::DenseArrayAttrImpl<U>>, std::string>
  convert(mlir::detail::DenseArrayAttrImpl<U> attr) {
    std::vector<std::string> result;
    for (auto element : attr.asArrayRef()) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
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
      if (auto denseBoolArrayAttr =
              mlir::dyn_cast<mlir::DenseBoolArrayAttr>(attr)) {
        return convert(denseBoolArrayAttr);
      }
      if (auto denseI8ArrayAttr =
              mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr)) {
        return convert(denseI8ArrayAttr);
      }
      if (auto denseI16ArrayAttr =
              mlir::dyn_cast<mlir::DenseI16ArrayAttr>(attr)) {
        return convert(denseI16ArrayAttr);
      }
      if (auto denseI32ArrayAttr =
              mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
        return convert(denseI32ArrayAttr);
      }
      if (auto denseI64ArrayAttr =
              mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
        return convert(denseI64ArrayAttr);
      }
      if (auto denseIntAttr =
              mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
        return convert(denseIntAttr);
      }
    }
    if constexpr (std::is_floating_point_v<T>) {
      if (auto denseF32ArrayAttr =
              mlir::dyn_cast<mlir::DenseF32ArrayAttr>(attr)) {
        return convert(denseF32ArrayAttr);
      }
      if (auto denseF64ArrayAttr =
              mlir::dyn_cast<mlir::DenseF64ArrayAttr>(attr)) {
        return convert(denseF64ArrayAttr);
      }
      if (auto denseFPAttr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(attr)) {
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

inline std::string convert(tt::DataType attr) {
  switch (attr) {
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
  case tt::DataType::Int32:
    return "::ttnn::DataType::INT32";
  case tt::DataType::Float16:
  case tt::DataType::BFP_Float2:
  case tt::DataType::BFP_Float4:
  case tt::DataType::BFP_Float8:
  case tt::DataType::BFP_BFloat2:
    llvm_unreachable("Unsupported ttnn::DataType");
  }

  llvm_unreachable("Unkonwn tt::DataType");
}

inline std::string convert(tt::DataTypeAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
  }

  return convert(attr.getValue());
}

inline std::string convert(ttnn::Layout attr) {
  switch (attr) {
  case ttnn::Layout::RowMajor:
    return "::ttnn::Layout::ROW_MAJOR";
  case ttnn::Layout::Tile:
    return "::ttnn::Layout::TILE";
  case ttnn::Layout::Invalid:
    return "::ttnn::Layout::INVALID";
  }

  llvm_unreachable("Unknown ttnn::Layout");
}

inline std::string convert(ttnn::LayoutAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
  }

  return convert(attr.getValue());
}

inline std::string convert(ttnn::TensorMemoryLayoutAttr attr) {
  if (!attr) {
    return TypeNameV<std::nullopt_t>;
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

  // TODO (azecevic): Add ShardSpec once it's modeled in the `MemoryConfigAttr`.
  std::string buf;
  llvm::raw_string_ostream rso(buf);
  rso << "::ttnn::MemoryConfig(";
  rso << convert(attr.getTensorMemoryLayout()) << ", ";
  rso << convert(attr.getBufferType());
  rso << ")";
  return buf;
}

template <typename TTNNOp>
class EmitCTTNNEmitter {
public:
  using OpAdaptor = typename TTNNOp::Adaptor;

  EmitCTTNNEmitter(TTNNOp op, OpAdaptor adaptor,
                   mlir::ConversionPatternRewriter &rewriter)
      : op{op}, adaptor{adaptor}, rewriter{rewriter} {}

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

  mlir::Attribute emit(tt::ttnn::MemoryConfigAttr attr) {
    return rewriter.getType<emitc::OpaqueAttr>(convert(attr));
  }

  template <typename TargetTy = void, typename SourceTy>
  mlir::Attribute emit(std::optional<SourceTy> attr) {
    if (!attr) {
      return rewriter.getType<emitc::OpaqueAttr>(TypeNameV<std::nullopt_t>);
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

  mlir::Attribute emit(Value val) {
    if (!val) {
      return rewriter.getType<emitc::OpaqueAttr>(TypeNameV<std::nullopt_t>);
    }

    auto operand =
        llvm::find_if(op->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.get() == val;
        });
    if (operand == op->getOpOperands().end()) {
      llvm_unreachable("Unknown operand");
    }

    return rewriter.getIndexAttr(operand->getOperandNumber());
  }

  // Handles the case when source type is convertible to mlir::Attribute type
  // and source and target types are in many-to-many relationship (i.e.
  // {mlir::ArrayAttr, mlir::DenseI32ArrayAttr} to {std::vector<uint32_t>,
  // ttnn::SmallVector<int32_t>}).
  template <typename TargetTy>
  mlir::Attribute emit(mlir::Attribute attr) {
    // It's assumed that the conversion might fail, in which case the result
    // will be `emitc::OpaqueAttr("::std::nullopt")`.
    if (auto convertedValue = EmitCTypeConverter<TargetTy>::convert(attr)) {
      return rewriter.getType<emitc::OpaqueAttr>(*convertedValue);
    }
    return rewriter.getType<emitc::OpaqueAttr>(TypeNameV<std::nullopt_t>);
  }

  // Handles the case when source type is a non mlir::Attribute convertible type
  // and source and target types are in many-to-many relationship (i.e.
  // llvm::ArrayRef<T> to std::vector<U>). For convenience, by default
  // `TargetTy` is the same as `SourceTy`, for cases where we already have an
  // appropriate C++ type.
  // TODO (azecevic): See if we can simplify the condition for this overload
  // instantiation.
  template <typename SourceTy, typename TargetTy = SourceTy>
  std::enable_if_t<!std::is_convertible_v<SourceTy, mlir::Attribute> &&
                       !std::is_convertible_v<SourceTy, mlir::Value>,
                   mlir::Attribute>
  emit(SourceTy &&attr) {
    auto result =
        EmitCTypeConverter<TargetTy>::convert(std::forward<SourceTy>(attr));
    // It's assumed that the conversion will always succeed, if the result is
    // `std::optional<std::string>` we assume that it contains the converted
    // value.
    if constexpr (std::is_same_v<decltype(result),
                                 std::optional<std::string>>) {
      return rewriter.getType<emitc::OpaqueAttr>(*result);
    }
    return rewriter.getType<emitc::OpaqueAttr>(result);
  }

  template <typename OpConversionPatternTy>
  emitc::CallOpaqueOp replaceOp(OpConversionPatternTy &&opConversionPattern,
                                llvm::ArrayRef<mlir::Attribute> args) {
    return rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op,
        opConversionPattern.getTypeConverter()->convertType(
            op->getResult(0).getType()),
        opConversionPattern.convertOpName(op), rewriter.getArrayAttr(args),
        nullptr, adaptor.getOperands());
  }

private:
  TTNNOp op;
  OpAdaptor adaptor;
  ConversionPatternRewriter &rewriter;
};

// Helper function that serves as an alternative to the
// `emit<std::variant<...>>` member function of the `EmitCTTNNEmitter` class.
// For example, instead of calling `emit<std::variant<int32_t, float>>(attr)`,
// one can call `emit<int32_t>(attr) | emit<float>(attr)`.
inline mlir::Attribute operator|(mlir::Attribute lhs, mlir::Attribute rhs) {
  static const mlir::Attribute nulloptAttr =
      emitc::OpaqueAttr::get(lhs.getContext(), TypeNameV<std::nullopt_t>);
  if (!lhs || lhs == nulloptAttr) {
    return rhs;
  }
  return lhs;
}

} // namespace ttnn_to_emitc
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
