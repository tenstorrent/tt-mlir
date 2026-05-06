// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_TRANSFORMUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_TRANSFORMUTILS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::tt::ttnn::utils {
// Get or insert device for the given operation.
GetDeviceOp getOrInsertDevice(mlir::RewriterBase &rewriter,
                              mlir::Operation *op);

GetDeviceOp getOrInsertDevice(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper method to insert a ToLayoutOp to convert the input operand to the
// desired tensor layout, buffer type and memory layout. When targetGrid is
// provided, the output encoding uses the given grid instead of deriving it
// from the input layout.
ToLayoutOp createToLayoutOp(
    mlir::Operation *op, mlir::TypedValue<RankedTensorType> inputValue,
    RewriterBase &rewriter, Layout targetTensorLayout,
    BufferType targetTensorBufferType,
    TensorMemoryLayoutAttr targetTensorMemoryLayout,
    ttcore::DataType targetTensorDataType, llvm::StringRef locSuffix = "",
    std::optional<llvm::ArrayRef<int64_t>> targetGridShape = std::nullopt);

} // namespace mlir::tt::ttnn::utils

#endif
