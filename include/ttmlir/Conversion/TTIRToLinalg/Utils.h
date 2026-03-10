// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H
#define TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>

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

} // namespace mlir::tt::ttir_to_linalg

#endif // TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H
