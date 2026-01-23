// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_UTILS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"

#include <type_traits>
#include <utility>

namespace mlir::tt::ttnn {
class DeviceType;
} // namespace mlir::tt::ttnn

namespace mlir::tt::ttir::utils {
namespace detail {
// It's assumed that operand is convertible to mlir::Value or mlir::ValueRange.
// The only exception being tt::ttnn::DeviceType, which is convertible to
// mlir::Value but should not be considered an operand.
template <typename T>
struct is_operand
    : std::bool_constant<
          (std::is_convertible_v<T, mlir::Value> ||
           std::is_convertible_v<T, mlir::ValueRange>)&&!std::
              is_convertible_v<T,
                               mlir::TypedValue<mlir::tt::ttnn::DeviceType>>> {
};

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
  static auto call(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value output, ArgsTy &&...args) {
    // Calls the generic build member function that every op has `static void
    // build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
    // ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
    // ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {})`.
    if constexpr (sizeof...(Js) == 0) {
      return builder.create<OpTy>(loc, output.getType(),
                                  ttmlir::utils::flatten<mlir::Value>(
                                      std::get<Is>(std::forward_as_tuple(
                                          std::forward<ArgsTy>(args)...))...,
                                      output));
    } else if constexpr (sizeof...(Js) == 1 &&
                         std::is_convertible_v<
                             std::tuple_element_t<
                                 sizeof...(Is) + sizeof...(Js) - 1,
                                 std::tuple<llvm::remove_cvref_t<ArgsTy>...>>,
                             mlir::ArrayRef<mlir::NamedAttribute>>) {
      return builder.create<OpTy>(
          loc, output.getType(),
          ttmlir::utils::flatten<mlir::Value>(
              std::get<Is>(
                  std::forward_as_tuple(std::forward<ArgsTy>(args)...))...,
              output),
          std::get<sizeof...(Is) + sizeof...(Js) - 1>(
              std::forward_as_tuple(std::forward<ArgsTy>(args)...)));
      // Otherwise, call the op specific builder that provides positional
      // `Attribute` arguments.
    } else {
      return builder.create<OpTy>(
          loc, output.getType(),
          std::get<Is>(std::forward_as_tuple(std::forward<ArgsTy>(args)...))...,
          output,
          std::get<sizeof...(Is) + Js>(
              std::forward_as_tuple(std::forward<ArgsTy>(args)...))...);
    }
  }
};

template <typename OpTy, size_t OperandCountV, size_t AttributeCountV>
struct SplitImpl {
  template <typename... ArgsTy>
  static auto call(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value dpsOutput, ArgsTy &&...args) {
    return SplitCaller<OpTy, std::make_index_sequence<OperandCountV>,
                       std::make_index_sequence<AttributeCountV>>::
        call(builder, loc, dpsOutput, std::forward<ArgsTy>(args)...);
  }
};

template <typename OpTy, typename... ArgsTy>
auto splitAndCall(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value output, ArgsTy &&...args) {
  constexpr size_t count = count_consecutive_v<ArgsTy...>;

  return SplitImpl<OpTy, count, sizeof...(ArgsTy) - count>::call(
      builder, loc, output, std::forward<ArgsTy>(args)...);
}
} // namespace detail

// Check if the given OpTy has the DestinationStyleOpInterface trait.
template <typename OpTy>
constexpr bool has_dps_trait_v =
    OpTy::template hasTrait<mlir::DestinationStyleOpInterface::Trait>();

// Wrapper for creating a DPS op with a given output type. It's assumed that a
// DPS op has exactly one output that comes after all of the inputs and before
// any of the attributes in the builder of an op. The output is generated using
// a ttir::EmptyOp. Calling this function:
// createDPSOp<OpTy>(rewriter, loc,  outputType, operand1, operand2, ...,
// operandN, attribute1, attribute2, ..., attributeM);
// is equivalent to:
// auto output = rewriter.create<ttir::EmptyOp>(loc, outputType.getShape(),
// outputType.getElementType(), outputType.getEncoding());
// rewriter.create<OpTy>(loc, outputType, operand1, operand2, ..., operandN,
// output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy createDPSOp(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::RankedTensorType outputType, ArgsTy &&...args) {
  static_assert(has_dps_trait_v<OpTy>);

  auto output = builder.create<mlir::tt::ttir::EmptyOp>(
      loc, outputType.getShape(), outputType.getElementType(),
      outputType.getEncoding());

  return detail::splitAndCall<OpTy>(builder, loc, output,
                                    std::forward<ArgsTy>(args)...);
}

// Wrapper for creating a DPS op with a given output shape, element type and
// encoding. It's assumed that a  DPS op has exactly one output that comes after
// all of the inputs and before any of the attributes in the builder of an op.
// The output is generated using a ttir::EmptyOp. Calling this function:
// createDPSOp<OpTy>(rewriter, loc,  outputShape, outputElementType,
// outputEncoding, operand1, operand2, ..., operandN, attribute1, attribute2,
// ..., attributeM);
// is equivalent to:
// auto outputType = mlir::RankedTensorType::get(outputShape, outputElementType,
// outputEncoding);
// auto output = rewriter.create<ttir::EmptyOp>(loc, outputShape,
// outputElementType, outputEncoding);
// rewriter.create<OpTy>(loc, outputType, operand1, operand2, ..., operandN,
// output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy createDPSOp(mlir::OpBuilder &builder, mlir::Location loc,
                 llvm::ArrayRef<int64_t> outputShape,
                 mlir::Type outputElementType, mlir::Attribute outputEncoding,
                 ArgsTy &&...args) {
  static_assert(has_dps_trait_v<OpTy>);

  auto outputType = mlir::RankedTensorType::get(outputShape, outputElementType,
                                                outputEncoding);
  return createDPSOp<OpTy>(builder, loc, outputType,
                           std::forward<ArgsTy>(args)...);
}

// Wrapper for replacing an op with a DPS op with a given output type.
// It's assumed that a  DPS op has exactly one output that comes after all of
// the inputs and before any of the attributes in the builder of a DPS op. The
// output is generated using a ttir::EmptyOp. Calling this function:
// replaceOpWithNewDPSOp<OpTy>(rewriter, op, outputType, operand1, operand2,
// ..., operandN, attribute1, attribute2, ..., attributeM);
// is equivalent to:
// auto output = rewriter.create<ttir::EmptyOp>(loc, outputType.getShape(),
// outputType.getElementType(), outputType.getEncoding());
// rewriter.replaceOpWithNewOp<OpTy>(op, outputType, operand1, operand2, ...,
// operandN, output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy replaceOpWithNewDPSOp(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           mlir::RankedTensorType outputType,
                           ArgsTy &&...args) {
  static_assert(has_dps_trait_v<OpTy>);

  auto newOp = createDPSOp<OpTy>(rewriter, op->getLoc(), outputType,
                                 std::forward<ArgsTy>(args)...);
  rewriter.replaceOp(op, newOp.getOperation());
  return newOp;
}

// Wrapper for replacing an op with a DPS op with a given output shape, element
// type and encoding. It's assumed that a  DPS op has exactly one output that
// comes after all of the inputs and before any of the attributes in the builder
// of a DPS op. The output is generated using a ttir::EmptyOp. Calling this
// function:
// replaceOpWithNewDPSOp<OpTy>(rewriter, op,  outputShape,
// outputElementType, outputEncoding, operand1, operand2, ..., operandN,
// attribute1, attribute2, ..., attributeM);
// is equivalent to:
// auto outputType = mlir::RankedTensorType::get(outputShape, outputElementType,
// outputEncoding);
// auto output = rewriter.create<ttir::EmptyOp>(loc, outputShape,
// outputElementType, outputEncoding);
// rewriter.replaceOpWithNewOp<OpTy>(op, outputType, operand1, operand2, ...,
// operandN, output, attribute1, attribute2, ..., attributeM);
template <typename OpTy, typename... ArgsTy>
OpTy replaceOpWithNewDPSOp(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                           llvm::ArrayRef<int64_t> outputShape,
                           mlir::Type outputElementType,
                           mlir::Attribute outputEncoding, ArgsTy &&...args) {
  static_assert(has_dps_trait_v<OpTy>);

  auto newOp =
      createDPSOp<OpTy>(rewriter, op->getLoc(), outputShape, outputElementType,
                        outputEncoding, std::forward<ArgsTy>(args)...);
  rewriter.replaceOp(op, newOp.getOperation());
  return newOp;
}

// Helper function to unsqueeze a value either on front or back dimension.
llvm::SmallVector<int64_t> unsqueezeValue(mlir::PatternRewriter &rewriter,
                                          mlir::Location loc,
                                          mlir::Value &input,
                                          mlir::RankedTensorType desiredType,
                                          bool frontUnsqueeze);

// Helper function to broadcast a value to desired shape.
mlir::LogicalResult broadcastValue(mlir::PatternRewriter &rewriter,
                                   mlir::Value input,
                                   mlir::RankedTensorType desiredType,
                                   mlir::Value &output, mlir::Location loc,
                                   bool frontUnsqueeze);

// Checks if an operation preserves a given dimension.
// For reshape, this checks both that the dimension size is unchanged and that
// the product of trailing dimensions (stride) is preserved.
// Dimension can be negative (counted from back, e.g. -1 for last dimension).
bool preservesDim(mlir::Operation *op, int64_t dim);

template <typename AdaptorT>
mlir::ValueRange getDpsInputsFromAdaptor(AdaptorT adaptor,
                                         unsigned numDpsInits) {
  const auto operands = adaptor.getOperands();
  assert(operands.size() >= numDpsInits &&
         "not enough operands for numDpsInits");
  return operands.drop_back(numDpsInits);
}

template <typename AdaptorT>
mlir::ValueRange getDpsOutputsFromAdaptor(AdaptorT adaptor,
                                          unsigned numDpsInits) {
  const auto operands = adaptor.getOperands();
  assert(operands.size() >= numDpsInits &&
         "not enough operands for numDpsInits");
  return operands.take_back(numDpsInits);
}

// Helper function to create a reshape operation.
inline ttir::ReshapeOp createReshapeOp(PatternRewriter &rewriter, Location loc,
                                       Value input,
                                       ::llvm::ArrayRef<int64_t> targetShape) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto shapeAttr =
      rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(targetShape));

  return rewriter.create<ttir::ReshapeOp>(
      loc,
      RankedTensorType::get(targetShape, inputType.getElementType(),
                            inputType.getEncoding()),
      input, shapeAttr);
}

// Helper function to flatten a tensor to 1D.
// It flattens tensor on given location with optional suffix to the location
// name.
inline Value flattenTensor(PatternRewriter &rewriter, Location loc,
                           Value tensor, const std::string &suffix = "") {
  RankedTensorType tensorType = mlir::cast<RankedTensorType>(tensor.getType());
  ArrayRef<int64_t> tensorShape = tensorType.getShape();

  // Calculate total number of elements (product of all dimensions).
  int64_t totalElements = 1;
  for (int64_t dim : tensorShape) {
    totalElements *= dim;
  }

  // Create new 1D shape.
  llvm::SmallVector<int64_t> flattenedShape = {totalElements};

  // Create location with optional suffix.
  Location reshapeLocation =
      suffix.empty() ? loc : ttmlir::utils::appendLocationSuffix(loc, suffix);

  // Reshape tensor to 1D.
  Value flattenedTensor =
      createReshapeOp(rewriter, reshapeLocation, tensor, flattenedShape);

  return flattenedTensor;
}

// Traces backward from a value through specified ops to find the source value.
template <typename... Ops>
mlir::Value lookThrough(mlir::Value value) {
  while (auto *op = value.getDefiningOp()) {
    if (llvm::isa<Ops...>(op)) {
      value = op->getOperand(0);
    } else {
      break;
    }
  }
  return value;
}

// Traces backward from a value through specified ops to find an operation of
// type OpTy.
template <typename OpTy, typename... Ops>
OpTy findOpThrough(mlir::Value value) {
  return lookThrough<Ops...>(value).template getDefiningOp<OpTy>();
}

// Traces backward through all layout ops (typecast, reshape, permute,
// broadcast, repeat_interleave) to find the source value.
inline mlir::Value lookThroughLayoutOps(mlir::Value value) {
  return lookThrough<TypecastOp, ReshapeOp, PermuteOp, BroadcastOp,
                     RepeatInterleaveOp>(value);
}

// Traces backward through all layout ops, but only looks through ops that
// satisfy the given predicate. The predicate receives the operation and should
// return true if we should look through it, false to stop.
template <typename PredicateFn>
mlir::Value lookThroughLayoutOpsIf(mlir::Value value, PredicateFn &&predicate) {
  while (auto *op = value.getDefiningOp()) {
    if (llvm::isa<TypecastOp, ReshapeOp, PermuteOp, BroadcastOp,
                  RepeatInterleaveOp>(op)) {
      if (!predicate(op)) {
        break;
      }
      value = op->getOperand(0);
    } else {
      break;
    }
  }
  return value;
}

// Traces backward through all layout ops to find an operation of type OpTy.
template <typename OpTy>
OpTy findOpThroughLayoutOps(mlir::Value value) {
  return lookThroughLayoutOps(value).template getDefiningOp<OpTy>();
}

// Traces backward through all layout ops that satisfy the predicate to find
// an operation of type OpTy.
template <typename OpTy, typename PredicateFn>
OpTy findOpThroughLayoutOpsIf(mlir::Value value, PredicateFn &&predicate) {
  return lookThroughLayoutOpsIf(value, std::forward<PredicateFn>(predicate))
      .template getDefiningOp<OpTy>();
}

// Moves the use-define chain of a value before a target operation.
// Used when operations are moved or new operations are added to ensure that
// operations are not placed before the definitions of their inputs, preserving
// MLIR's topological ordering.
inline void moveUDChainBefore(mlir::Value value, mlir::Operation *targetOp) {
  if (!value.getDefiningOp() ||
      value.getDefiningOp()->isBeforeInBlock(targetOp)) {
    return;
  }

  llvm::SetVector<mlir::Value> udChain = ttmlir::utils::getUseDefChain(value);
  llvm::SetVector<mlir::Operation *> udChainOps =
      ttmlir::utils::filterOperations(udChain.getArrayRef());
  llvm::SetVector<mlir::Operation *> udChainSorted =
      mlir::topologicalSort(udChainOps);

  // We are not moving ops in UD chain that are already before the target, as
  // they could have descendants that are also before target but are not in
  // the UD chain.
  for (auto *op : udChainSorted) {
    if (op->isBeforeInBlock(targetOp)) {
      continue;
    }
    op->moveBefore(targetOp);
  }
}

} // namespace mlir::tt::ttir::utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_UTILS_H
