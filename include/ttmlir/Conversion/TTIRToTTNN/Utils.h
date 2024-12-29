// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTNN_UTILS_H
#define TTMLIR_CONVERSION_TTIRTOTTNN_UTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttir_to_ttnn::utils {

// Given a 4D tensor, flatten the first 3 dimensions into the 3rd dimension and
// keep the 4th dimension as is.
inline SmallVector<int64_t, 4> flattenNHW(ArrayRef<int64_t> shape) {
  assert(shape.size() == 4 && "Expected 4D tensor");
  return {1, 1, shape[0] * shape[1] * shape[2], shape[3]};
}

// Generates a reshape operation for the given input tensor with the new shape.
mlir::tt::ttnn::ReshapeOp generateReshape(mlir::Value input,
                                          llvm::ArrayRef<int64_t> newShape,
                                          mlir::PatternRewriter &rewriter);

// Generates a reshape operation for the given input tensor that returns 4D
// tensor. Assumes that the input tensor is 4D. First 3 dimensions are flattened
// into 3rd dimension and 4th dimension is kept as is.
mlir::tt::ttnn::ReshapeOp generateNHWFlatten(mlir::Value input,
                                             mlir::PatternRewriter &rewriter);

} // namespace mlir::tt::ttir_to_ttnn::utils

#endif
