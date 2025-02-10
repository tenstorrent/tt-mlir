// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
#define TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "ttmlir/Conversion/TTNNToEmitC/Utils.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsDialect.h.inc"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include <array>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Transforms/DialectConversion.h>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ttnn {
template <typename T>
class SmallVector;
} // namespace ttnn

namespace mlir {
namespace tt {
namespace ttnn_to_emitc {

template <typename T>
struct EmitCSerializer;

template <typename T>
const std::string EmitCSerializerV = EmitCSerializer<T>::value;

template <>
struct EmitCSerializer<int32_t> {
  inline static const std::string value = "int32_t";

  static std::string convert(int32_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<int64_t> {
  inline static const std::string value = "int64_t";

  static std::string convert(int64_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<uint32_t> {
  inline static const std::string value = "uint32_t";

  static std::string convert(uint32_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<uint64_t> {
  static constexpr std::string_view value = "uint64_t";

  static std::string convert(uint64_t value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<float> {
  inline static const std::string value = "float";

  static std::string convert(float value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<double> {
  inline static const std::string value = "double";

  static std::string convert(double value) { return std::to_string(value); }
};

template <>
struct EmitCSerializer<bool> {
  inline static const std::string value = "bool";

  static std::string convert(bool value) { return value ? "true" : "false"; }
};

template <>
struct EmitCSerializer<std::string> {
  inline static const std::string value = "::std::string";

  static std::string convert(std::string value) { return "\"" + value + "\""; }
};

template <typename T, size_t k>
struct EmitCSerializer<std::array<T, k>> {
  inline static const std::string value =
      "::std::array<" + EmitCSerializerV<T> + ", " + std::to_string(k) + ">";

  static std::string convert(std::array<T, k> val) {
    return std::string(EmitCSerializer<std::array<T, k>>::value);
  }
};

template <typename T>
struct EmitCSerializer<std::vector<T>> {
  inline static const std::string value =
      "::std::vector<" + EmitCSerializerV<T> + ">";

  static std::string convert(std::vector<std::string> val) {
    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << EmitCSerializer<std::vector<T>>::value << "{";
    llvm::interleaveComma(val, rso);
    rso << "}";
    return buf;
  }
};

template <typename T, typename = void>
struct EmitCTypeConverter;

// Converter for integral types.
template <typename T>
struct EmitCTypeConverter<T, std::enable_if_t<std::is_integral_v<T>, void>> {
  using type = T;

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
    if constexpr (std::is_signed_v<type>) {
      return convert(value.getSExtValue());
    }
    return convert(value.getZExtValue());
  }

  template <typename U, std::enable_if_t<std::is_integral_v<U>, bool> = true>
  static std::string convert(U value) {
    return EmitCSerializer<type>::convert(static_cast<type>(value));
  }
};

// Converter for floating point types.
template <typename T>
struct EmitCTypeConverter<T,
                          std::enable_if_t<std::is_floating_point_v<T>, void>> {
  using type = T;

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
    return EmitCSerializer<type>(static_cast<type>(value));
  }
};

// Convert container types (std::vector, ttnn::SmallVector, etc.).
template <typename T>
struct EmitCContainerTypeConverter {
  using type = T;
  using value_type = typename type::value_type;

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
    return EmitCSerializer<type>::convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return EmitCSerializer<type>::convert(result);
  }

  static std::string convert(mlir::DenseFPElementsAttr attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return EmitCSerializer<type>::convert(result);
  }

  template <typename U>
  static std::string convert(llvm::ArrayRef<U> attr) {
    std::vector<std::string> result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<value_type>::convert(element));
    }
    return EmitCSerializer<type>::convert(result);
  }
};

// TODO (azecevic): EmitCContainerTypeConverter for ttnn::SmallVector,
// ttnn::span...

template <typename T>
struct EmitCTypeConverter<std::vector<T>>
    : public EmitCContainerTypeConverter<std::vector<T>> {};

template <typename T>
struct EmitCTypeConverter<::ttnn::SmallVector<T>>
    : public EmitCContainerTypeConverter<::ttnn::SmallVector<T>> {};

template <typename T, size_t k>
struct EmitCTypeConverter<std::array<T, k>> {
  using type = std::array<T, k>;

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
    type result;
    for (size_t i = 0; i < attr.size(); ++i) {
      auto element = EmitCTypeConverter<T>::convert(attr[i]);
      if (!element) {
        return {};
      }
      result[i] = *element;
    }
    return EmitCSerializer<type>::convert(result);
  }

  static std::string convert(mlir::DenseIntElementsAttr attr) {
    assert(attr.size() == k &&
           "DenseIntElementsAttr size does not match std::array size");
    type result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return EmitCSerializer<type>::convert(result);
  }

  static type convert(mlir::DenseFPElementsAttr attr) {
    assert(attr.size() == k &&
           "DenseFPElementsAttr size does not match std::array size");
    type result;
    for (int64_t i = 0; i < attr.size(); ++i) {
      result[i] = EmitCTypeConverter<T>::convert(*(attr.begin() + i));
    }
    return EmitCSerializer<type>::convert(result);
  }

  template <typename U>
  static type convert(llvm::ArrayRef<U> attr) {
    assert(attr.size() == k && "ArrayRef size does not match std::array size");
    type result;
    for (auto element : attr) {
      result.push_back(EmitCTypeConverter<T>::convert(element));
    }
    return EmitCSerializer<type>::convert(result);
  }
};

// TODO (azecevic): Try to fix this!
// template <typename First, typename... Rest>
// struct EmitCTypeConverter<std::variant<First, Rest...>> {
//   using type = std::variant<First, Rest...>;

//   static std::string convert(mlir::Attribute attr) {
//     if (auto tryFirst = EmitCTypeConverter<First>::convert(attr)) {
//       return *tryFirst;
//     }
//     if constexpr (sizeof...(Rest) > 0) {
//       auto rest = EmitCTypeConverter<std::variant<Rest...>>::convert(attr);
//       return std::visit([](auto &&val) -> std::string { return *val; },
//       rest);
//     }
//     llvm_unreachable("Invalid variant type");
//     return {};
//   }
// };

template <>
struct EmitCTypeConverter<std::nullopt_t> {
  using type = std::nullopt_t;

  static std::string convert(mlir::Attribute attr) {
    assert(!attr && "Not an optional attribute");
    return "::std::nullopt";
  }

  static std::string convert(std::nullopt_t) { return "::std::nullopt"; }
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

template <typename TTNNOp>
class EmitCTTNNEmitter {
public:
  using OpAdaptor = typename TTNNOp::Adaptor;

  EmitCTTNNEmitter(ConversionPatternRewriter &rewriter, TTNNOp op,
                   OpAdaptor adaptor)
      : rewriter{rewriter}, op{op}, adaptor{adaptor} {}

  mlir::Attribute operator()(tt::ttnn::ShapeAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(
        tt::ttnn_to_emitc::convert(attr));
  }

  mlir::Attribute operator()(tt::DataTypeAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(
        tt::ttnn_to_emitc::convert(attr));
  }

  mlir::Attribute operator()(tt::ttnn::LayoutAttr attr) {
    return rewriter.getAttr<emitc::OpaqueAttr>(
        tt::ttnn_to_emitc::convert(attr));
  }

  mlir::Attribute operator()(tt::ttnn::MemoryConfigAttr attr) {
    if (!attr) {
      return rewriter.getType<emitc::OpaqueAttr>("::std::nullopt");
    }
    // TODO (azecevic): Implement this.
    return rewriter.getAttr<emitc::OpaqueAttr>("::std::nullopt");
  }

  template <typename T>
  mlir::Attribute operator()(std::optional<T> attr) {
    if (!attr) {
      return rewriter.getType<emitc::OpaqueAttr>("::std::nullopt");
    }

    return operator()(*attr);
  }

  mlir::Attribute operator()(std::nullopt_t) {
    return rewriter.getType<emitc::OpaqueAttr>("::std::nullopt");
  }

  mlir::Attribute operator()(bool attr) {
    return rewriter.getType<emitc::OpaqueAttr>(attr ? "true" : "false");
  }

  mlir::Attribute operator()(Value val) {
    if (!val) {
      return rewriter.getType<emitc::OpaqueAttr>("::std::nullopt");
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

  template <typename T>
  mlir::Attribute operator()(mlir::Attribute attr) {
    if (auto convertedValue =
            tt::ttnn_to_emitc::EmitCTypeConverter<T>::convert(attr)) {
      return rewriter.getType<emitc::OpaqueAttr>(*convertedValue);
    }
    return {};
  }

private:
  ConversionPatternRewriter &rewriter;
  TTNNOp op;
  OpAdaptor adaptor;
};

} // namespace ttnn_to_emitc
} // namespace tt
} // namespace mlir

#endif // TTMLIR_CONVERSION_TTNNTOEMITC_EMITCCONVERSION_H
