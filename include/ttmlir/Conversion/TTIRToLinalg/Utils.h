// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H
#define TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <cstdint>
#include <functional>

// Forward-declare to avoid pulling in the full TTIROps header.
namespace mlir::tt::ttir {
class FlattenedCompatInfoAttr;
} // namespace mlir::tt::ttir

namespace mlir::tt::ttir_to_linalg {

//===----------------------------------------------------------------------===//
// Broadcasting helpers
//===----------------------------------------------------------------------===//

// Get the dimensions to broadcast.
//
// This function calculates the dimensions to broadcast. We assume that input
// and target shapes are broadcastable. For example if input shape is [4, 1, 3]
// and we want to broadcast to [1, 4, 5, 3], the function will return [0, 2]
// since we want to broadcast 0th and 2nd dimension of result shape.
SmallVector<int64_t, 2> getBroadcastDims(ArrayRef<int64_t> inputShape,
                                         ArrayRef<int64_t> targetShape);

// Get the dimensions to collapse.
//
// This function calculates the dimensions to collapse. We assume that input
// and target shapes are broadcastable. linalg.broadcast requires that input
// tensor only contains dimensions that won't be broadcasted in input tensor.
// For example if input shape is [4, 1, 3] and we want to broadcast to [4, 5,
// 3], then we need to collapse the first dimension of input tensor to [4, 3].
// This function calculates the dimensions to collapse. In case above, we will
// return [[0], [1, 2]].
SmallVector<SmallVector<int64_t, 2>, 2>
getCollapseDims(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> targetShape);

// Broadcast input to targetShape using linalg.broadcast if needed.
// Returns the input unchanged if shapes already match.
Value broadcastToShape(Value input, ArrayRef<int64_t> targetShape, Location loc,
                       ConversionPatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Tensor/TOSA helpers
//===----------------------------------------------------------------------===//

// Convert a tensor of numeric (integer or floating-point) values to a tensor
// of boolean values by comparing with zero; zero is false and nonzero is true.
Value convertToBooleanTensor(Value input, Location loc,
                             ConversionPatternRewriter &rewriter);

// Helper function to create DenseElementsAttr with a specific value based on
// element type.
DenseElementsAttr createDenseElementsAttr(RankedTensorType resultType,
                                          double value);

// Helper to create a ranked TOSA constant with shape [1, 1, ...] matching rank.
Value createTosaConst(ConversionPatternRewriter &rewriter, Location loc,
                      Type elementType, int64_t rank, double value);

// Helper to create the TOSA mul shift operand (i8 zero tensor).
Value createTosaMulShift(ConversionPatternRewriter &rewriter, Location loc);

//===----------------------------------------------------------------------===//
// Dimension helpers
//===----------------------------------------------------------------------===//

// Normalize negative dimension to positive. Negative dimensions are interpreted
// as indexing from the end (e.g., -1 is the last dimension).
int64_t normalizeDim(int64_t dim, int64_t rank);

//===----------------------------------------------------------------------===//
// TOSA reduction helpers
//===----------------------------------------------------------------------===//

// Create a chain of TOSA reduction operations for multiple dimensions.
// Reduces each dimension in descending order and optionally reshapes if
// !keepDim.
template <typename TosaReductionOp>
Value createReductionOpChain(Value input, RankedTensorType resultType,
                             ArrayRef<int64_t> dims, bool keepDim, Location loc,
                             ConversionPatternRewriter &rewriter) {
  Value result = input;
  auto inputType = cast<RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();

  // Normalize negative dims and sort in descending order.
  SmallVector<int64_t> normalizedDims(dims.begin(), dims.end());
  for (int64_t &d : normalizedDims) {
    d = normalizeDim(d, rank);
  }
  std::sort(normalizedDims.begin(), normalizedDims.end(),
            std::greater<int64_t>());

  SmallVector<int64_t> shape(inputType.getShape().begin(),
                             inputType.getShape().end());
  for (size_t i = 0; i < normalizedDims.size(); ++i) {
    int64_t dim = normalizedDims[i];
    auto axisAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(dim));
    shape[dim] = 1;
    auto opResultType =
        RankedTensorType::get(shape, inputType.getElementType());
    result =
        rewriter.create<TosaReductionOp>(loc, opResultType, result, axisAttr);
  }
  if (!keepDim) {
    ArrayRef<int64_t> newShape = resultType.getShape();
    auto shapeType =
        tosa::shapeType::get(rewriter.getContext(), newShape.size());
    auto attr = rewriter.getIndexTensorAttr(newShape);
    auto shapeOp = rewriter.create<tosa::ConstShapeOp>(loc, shapeType, attr);
    result = rewriter.create<tosa::ReshapeOp>(loc, resultType, result, shapeOp);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Sliding-window / pooling helpers
//===----------------------------------------------------------------------===//

// Calculate extra padding needed for ceil_mode in pooling/conv ops.
int64_t calculateExtraPadding(int64_t dim, int64_t kernel, int64_t stride,
                              int64_t padding1, int64_t padding2,
                              int64_t dilation);

// Create a tosa.reshape to the given target type.
Value createTosaReshape(Value input, RankedTensorType targetType,
                        ConversionPatternRewriter &rewriter, Location loc);

// Unflatten input from (1, 1, N*H*W, C) to (N, H, W, C) using metadata from
// FlattenedCompatInfoAttr.
Value unflattenInput(Value input, ttir::FlattenedCompatInfoAttr flatInfo,
                     ConversionPatternRewriter &rewriter, Location loc);

// Slice a result tensor back to a target shape if they differ. Handles cases
// where extra padding caused the computed output to be larger than expected.
Value sliceResultToShape(Value result, RankedTensorType targetType,
                         ConversionPatternRewriter &rewriter, Location loc);

} // namespace mlir::tt::ttir_to_linalg

#endif // TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H
