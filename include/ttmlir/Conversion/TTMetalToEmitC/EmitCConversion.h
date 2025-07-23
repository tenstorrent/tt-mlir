// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTMETALTOEMITC_EMITCCONVERSION_H
#define TTMLIR_CONVERSION_TTMETALTOEMITC_EMITCCONVERSION_H

#include "ttmlir/Conversion/TTMetalToEmitC/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// Mock definitions of tt-metal types for conversion purposes
namespace tt {
namespace tt_metal {
struct Device;
struct Buffer;
struct Program;
struct CommandQueue;
struct ComputeConfig;
struct DataMovementConfig;
struct EthernetConfig;

// Core coordination types
struct CoreCoord {
  std::size_t x, y;
};

struct CoreRange {
  CoreCoord start, end;
};

struct CoreRangeSet {
  std::vector<CoreRange> ranges;
};

} // namespace tt_metal
} // namespace tt

namespace mlir::tt::ttmetal_to_emitc {

template <typename T>
struct TypeName;

template <typename... Types>
struct JoinTypeNames;

template <typename T>
const std::string TypeNameV = TypeName<T>::value;

template <typename... Types>
const std::string JoinTypeNamesV = JoinTypeNames<Types...>::value;

// Basic type names
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
struct TypeName<bool> {
  inline static const std::string value = "bool";
};

template <>
struct TypeName<std::string> {
  inline static const std::string value = "::std::string";
};

template <>
struct TypeName<std::nullopt_t> {
  inline static const std::string value = "::std::nullopt";
};

// tt-metal type names
template <>
struct TypeName<::tt::tt_metal::Device> {
  inline static const std::string value = "::tt::tt_metal::Device";
};

template <>
struct TypeName<::tt::tt_metal::Buffer> {
  inline static const std::string value = "::tt::tt_metal::Buffer";
};

template <>
struct TypeName<::tt::tt_metal::Program> {
  inline static const std::string value = "::tt::tt_metal::Program";
};

template <>
struct TypeName<::tt::tt_metal::CommandQueue> {
  inline static const std::string value = "::tt::tt_metal::CommandQueue";
};

template <>
struct TypeName<::tt::tt_metal::CoreCoord> {
  inline static const std::string value = "::tt::tt_metal::CoreCoord";
};

template <>
struct TypeName<::tt::tt_metal::CoreRange> {
  inline static const std::string value = "::tt::tt_metal::CoreRange";
};

template <>
struct TypeName<::tt::tt_metal::CoreRangeSet> {
  inline static const std::string value = "::tt::tt_metal::CoreRangeSet";
};

template <>
struct TypeName<::tt::tt_metal::ComputeConfig> {
  inline static const std::string value = "::tt::tt_metal::ComputeConfig";
};

template <>
struct TypeName<::tt::tt_metal::DataMovementConfig> {
  inline static const std::string value = "::tt::tt_metal::DataMovementConfig";
};

template <>
struct TypeName<::tt::tt_metal::EthernetConfig> {
  inline static const std::string value = "::tt::tt_metal::EthernetConfig";
};

// Container types
template <typename T>
struct TypeName<std::vector<T>> {
  inline static const std::string value = "::std::vector<" + TypeNameV<T> + ">";
};

template <typename T>
struct TypeName<std::optional<T>> {
  inline static const std::string value = "::std::optional<" + TypeNameV<T> + ">";
};

// Type name joining for template parameters
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

// Type converters for MLIR attributes to C++ values
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

// Converter for integral types
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
      return std::to_string(value.getSExtValue());
    }
    return std::to_string(value.getZExtValue());
  }

  template <typename U>
  static std::enable_if_t<std::is_integral_v<U>, std::string> convert(U value) {
    return std::to_string(static_cast<T>(value));
  }
};

// CoreRange converter - simplified for now
template <>
struct EmitCTypeConverter<::tt::tt_metal::CoreRange> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    // TODO: Implement proper CoreRange attribute conversion
    return "tt::tt_metal::CoreRange{{0, 0}, {1, 1}}";
  }
};

// ComputeConfig converter - simplified for now
template <>
struct EmitCTypeConverter<::tt::tt_metal::ComputeConfig> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    // TODO: Implement proper ComputeConfig attribute conversion
    return "tt::tt_metal::ComputeConfig{}";
  }
};

// Vector converter for container types
template <typename T>
struct EmitCTypeConverter<std::vector<T>> {
  static std::optional<std::string> convert(mlir::Attribute attr) {
    if (!attr) {
      return {};
    }

    if (auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      return convert(arrayAttr);
    }
    return {};
  }

  static std::string convert(mlir::ArrayAttr attr) {
    llvm::SmallVector<std::string> result;
    for (auto element : attr) {
      auto converted = EmitCTypeConverter<T>::convert(element);
      if (converted) {
        result.push_back(*converted);
      }
    }

    std::string buf;
    llvm::raw_string_ostream rso(buf);
    rso << TypeNameV<std::vector<T>> << "{";
    llvm::interleaveComma(result, rso);
    rso << "}";
    return buf;
  }
};

// TTMetal operation emitter
template <typename TTMetalOp>
class EmitCTTMetalEmitter {
public:
  using OpAdaptor = typename TTMetalOp::Adaptor;

  EmitCTTMetalEmitter(TTMetalOp op, OpAdaptor adaptor,
                      mlir::ConversionPatternRewriter &rewriter)
      : op{op}, adaptor{adaptor}, rewriter{rewriter} {}

  EmitCTTMetalEmitter(const EmitCTTMetalEmitter &) = delete;
  EmitCTTMetalEmitter &operator=(const EmitCTTMetalEmitter &) = delete;

  // Emit MLIR value as operand index
  mlir::Attribute emit(mlir::Value val) {
    if (!val) {
      return emit(std::nullopt);
    }

    unsigned index = getOperandIndex(val);
    operands.push_back(adaptor.getOperands()[index]);
    return rewriter.getIndexAttr(index);
  }

  // Emit nullopt
  mlir::Attribute emit(std::nullopt_t) {
    return rewriter.getAttr<emitc::OpaqueAttr>(TypeNameV<std::nullopt_t>);
  }

  // Emit MLIR attribute using type converter
  template <typename TargetTy>
  mlir::Attribute emit(mlir::Attribute attr) {
    auto convertedValue = EmitCTypeConverter<TargetTy>::convert(attr);
    if (convertedValue) {
      return rewriter.getAttr<emitc::OpaqueAttr>(*convertedValue);
    }
    return emit(std::nullopt);
  }

  // Replace operation with EmitC call
  template <typename OpConversionPatternTy>
  mlir::Value replaceOp(OpConversionPatternTy &&opConversionPattern,
                        llvm::ArrayRef<mlir::Attribute> args) {
    auto resultTypes = llvm::to_vector(
        llvm::map_range(op->getResultTypes(), [&](Type type) -> Type {
          return opConversionPattern.getTypeConverter()->convertType(type);
        }));

    auto callOpaqueOp = rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, resultTypes, opConversionPattern.convertOpName(op),
        rewriter.getArrayAttr(args), /*template_args=*/nullptr, operands);

    if (callOpaqueOp.getNumResults() == 0) {
      return {};
    }
    return callOpaqueOp.getResult(0);
  }

private:
  unsigned getOperandIndex(mlir::Value value) {
    auto *opOperand = std::find_if(
        op->getOpOperands().begin(), op->getOpOperands().end(),
        [&](mlir::OpOperand &operand) { return operand.get() == value; });

    return opOperand->getOperandNumber();
  }

  TTMetalOp op;
  OpAdaptor adaptor;
  ConversionPatternRewriter &rewriter;
  llvm::SmallVector<mlir::Value> operands;
};

} // namespace mlir::tt::ttmetal_to_emitc

#endif // TTMLIR_CONVERSION_TTMETALTOEMITC_EMITCCONVERSION_H