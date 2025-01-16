// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_UTILS_H
#define TTMLIR_UTILS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

// Returns a string that is the concatenation of the string representations of
// Range R elements interleaved with separator. Example: join({1, 2, 3}, ", ")
// -> "1, 2, 3"
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

// Returns a vector `broadcastShape`, such that each index i of inputShape
// multiplied by index i of broadcastShape is equal to index i of outputShape.
// Example:  inputShape = [1, 32, 1], outputShape = [1, 32, 16] -> [1, 1, 16]
template <typename T>
inline llvm::SmallVector<T>
getBroadcastDimensions(llvm::ArrayRef<int64_t> inputShape,
                       llvm::ArrayRef<int64_t> outputShape) {
  assert(inputShape.size() == outputShape.size() &&
         "Input and Output shape should match.");

  llvm::SmallVector<T> broadcastShape;
  for (size_t i = 0; i < outputShape.size(); i++) {
    T d = outputShape[i] / inputShape[i];
    broadcastShape.push_back(d);
  }

  return broadcastShape;
}

// For a given llvm::APInt value, returns it as a C++ integer type T.
template <typename T>
inline T integerAs(const llvm::APInt &value) {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(value.getSExtValue());
  } else {
    static_assert(std::is_unsigned_v<T>,
                  "T must be signed or unsigned integer type");
    return static_cast<T>(value.getZExtValue());
  }
}

// For a given mlir::Attribute attr, returns a pair of integers of type
// ReturnTy. If attr is an IntegerAttr, it's interpreted as a (value(attr),
// value(attr)) pair of values, where value(attr) is of type ScalarTy. If attr
// is a DenseArrayAttr<VectorElementTy> of size 2, it's interpreted as a
// (attr[0], attr[1]) pair of values. Otherwise, returns an error message.
template <typename ScalarTy, typename VectorElementTy = ScalarTy,
          typename ReturnTy = ScalarTy>
inline llvm::Expected<std::pair<ReturnTy, ReturnTy>>
getPairOfInteger(mlir::Attribute attr) {
  ReturnTy x{};
  ReturnTy y{};
  // If attr is IntgerAttr, it's interpreted as a (attr, attr) pair of values.
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    x = y = integerAs<ScalarTy>(value.getValue());
    // If attr is DenseArrayAttr, it's interpreted as a (attr[0], attr[1]) pair
    // of values if it has size 2.
  } else if (auto tuple = mlir::dyn_cast<
                 ::mlir::detail::DenseArrayAttrImpl<VectorElementTy>>(attr);
             tuple.size() == 2) {
    x = tuple[0];
    y = tuple[1];
    // Otherwise, it's an error.
  } else if (tuple) {
    return llvm::createStringError(
        "Expected integer or pair of integers, got tuple of size %lu",
        tuple.size());
  } else {
    return llvm::createStringError("Unexpected attribute type");
  }

  return std::make_pair(x, y);
}

// It's assumed that operand is convertible to mlir::Value or mlir::ValueRange.
// The only exception being tt::DeviceType, which is convertible to mlir::Value
// but should not be considered an operand.
template <typename T>
struct is_operand
    : std::bool_constant<
          (std::is_convertible_v<T, mlir::Value> ||
           std::is_convertible_v<T, mlir::ValueRange>) &&
          !std::is_convertible_v<T, mlir::TypedValue<mlir::tt::DeviceType>>> {};

template <typename T>
inline constexpr bool is_operand_v = is_operand<T>::value;

template <typename... ArgsTy>
struct count_consecutive : std::integral_constant<size_t, 0> {};

template <typename... ArgsTy>
inline constexpr size_t count_consecutive_v =
    count_consecutive<ArgsTy...>::value;

template <typename FirstTy, typename... RestTy>
struct count_consecutive<FirstTy, RestTy...>
    : std::conditional_t<
          is_operand_v<FirstTy>,
          std::integral_constant<size_t, 1 + count_consecutive_v<RestTy...>>,
          std::integral_constant<size_t, 0>> {};

template <typename OpTy, typename IndexSeqFirst, typename IndexSeqRest>
struct SplitCaller;

template <typename OpTy, size_t... Is, size_t... Js>
struct SplitCaller<OpTy, std::index_sequence<Is...>,
                   std::index_sequence<Js...>> {
  template <typename... ArgsTy>
  static auto call(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value output, ArgsTy &&...args) {
    return rewriter.create<OpTy>(
        loc, output.getType(),
        std::get<Is>(std::forward_as_tuple(std::forward<ArgsTy>(args)...))...,
        output,
        std::get<sizeof...(Is) + Js>(
            std::forward_as_tuple(std::forward<ArgsTy>(args)...))...);
  }
};

template <typename OpTy, size_t OperandCountV, size_t AttributeCountV>
struct SplitImpl {
  template <typename... ArgsTy>
  static auto call(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value dpsOutput, ArgsTy &&...args) {
    return SplitCaller<OpTy, std::make_index_sequence<OperandCountV>,
                       std::make_index_sequence<AttributeCountV>>::
        call(rewriter, loc, dpsOutput, std::forward<ArgsTy>(args)...);
  }
};

template <typename OpTy, typename... ArgsTy>
auto splitAndCall(mlir::PatternRewriter &rewriter, mlir::Location loc,
                  mlir::Value output, ArgsTy &&...args) {
  constexpr size_t count = count_consecutive_v<ArgsTy...>;

  return SplitImpl<OpTy, count, sizeof...(ArgsTy) - count>::call(
      rewriter, loc, output, std::forward<ArgsTy>(args)...);
}

// Wrapper for creating a DPS op with a given output type. It's assumed that a
// DPS op has exactly one output that comes after all of the inputs and before
// any of the attributes in the builder of an op. The output is generated using
// a tensor::EmptyOp. Calling this function:
// createDPSOp<OpTy>(rewriter, loc,  outputType, operand1, operand2, ...,
// operandN, attribute1, attribute2, ..., attributeM);
// is equivalent to:
// auto output = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(),
// outputType.getElementType(), outputType.getEncoding());
// rewriter.create<OpTy>(loc, outputType, operand1, operand2, ..., operandN,
// output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy createDPSOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::RankedTensorType outputType, ArgsTy &&...args) {
  auto output = rewriter.create<mlir::tensor::EmptyOp>(
      loc, outputType.getShape(), outputType.getElementType(),
      outputType.getEncoding());

  return splitAndCall<OpTy>(rewriter, loc, output,
                            std::forward<ArgsTy>(args)...);
}

// Wrapper for creating a DPS op with a given output shape, element type and
// encoding. It's assumed that a  DPS op has exactly one output that comes after
// all of the inputs and before any of the attributes in the builder of an op.
// The output is generated using a tensor::EmptyOp. Calling this function:
// createDPSOp<OpTy>(rewriter, loc,  outputShape, outputElementType,
// outputEncoding, operand1, operand2, ..., operandN, attribute1, attribute2,
// ..., attributeM);
// is equivalent to:
// auto outputType = mlir::RankedTensorType::get(outputShape, outputElementType,
// outputEncoding);
// auto output = rewriter.create<tensor::EmptyOp>(loc, outputShape,
// outputElementType, outputEncoding);
// rewriter.create<OpTy>(loc, outputType, operand1, operand2, ..., operandN,
// output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy createDPSOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::Type outputElementType, mlir::Attribute outputEncoding,
                 ArgsTy &&...args) {
  auto outputType = mlir::RankedTensorType::get(outputShape, outputElementType,
                                                outputEncoding);
  return createDPSOp<OpTy>(rewriter, loc, outputType,
                           std::forward<ArgsTy>(args)...);
}

// Wrapper for replacing an op with a DPS op with a given output type.
// It's assumed that a  DPS op has exactly one output that comes after all of
// the inputs and before any of the attributes in the builder of a DPS op. The
// output is generated using a tensor::EmptyOp. Calling this function:
// replaceOpWithNewDPSOp<OpTy>(rewriter, op, outputType, operand1, operand2,
// ..., operandN, attribute1, attribute2, ..., attributeM);
// is equivalent to:
// auto output = rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(),
// outputType.getElementType(), outputType.getEncoding());
// rewriter.replaceOpWithNewOp<OpTy>(op, outputType, operand1, operand2, ...,
// operandN, output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy replaceOpWithNewDPSOp(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           mlir::RankedTensorType outputType,
                           ArgsTy &&...args) {
  auto newOp = createDPSOp<OpTy>(rewriter, op->getLoc(), outputType,
                                 std::forward<ArgsTy>(args)...);
  rewriter.replaceOp(op, newOp.getOperation());
  return newOp;
}

// Wrapper for replacing an op with a DPS op with a given output shape, element
// type and encoding. It's assumed that a  DPS op has exactly one output that
// comes after all of the inputs and before any of the attributes in the builder
// of a DPS op. The output is generated using a tensor::EmptyOp. Calling this
// function:
// replaceOpWithNewDPSOp<OpTy>(rewriter, op,  outputShape,
// outputElementType, outputEncoding, operand1, operand2, ..., operandN,
// attribute1, attribute2, ..., attributeM);
// is equivalent to:
// auto outputType = mlir::RankedTensorType::get(outputShape, outputElementType,
// outputEncoding);
// auto output = rewriter.create<tensor::EmptyOp>(loc, outputShape,
// outputElementType, outputEncoding);
// rewriter.replaceOpWithNewOp<OpTy>(op, outputType, operand1, operand2, ...,
// operandN, output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy replaceOpWithNewDPSOp(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           llvm::ArrayRef<int64_t> outputShape,
                           mlir::Type outputElementType,
                           mlir::Attribute outputEncoding, ArgsTy &&...args) {
  auto newOp =
      createDPSOp<OpTy>(rewriter, op->getLoc(), outputShape, outputElementType,
                        outputEncoding, std::forward<ArgsTy>(args)...);
  rewriter.replaceOp(op, newOp.getOperation());
  return newOp;
}

} // namespace ttmlir::utils

#endif
