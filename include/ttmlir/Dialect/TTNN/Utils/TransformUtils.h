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
// desired tensor layout, buffer type and memory layout.
ToLayoutOp createToLayoutOp(mlir::Operation *op,
                            mlir::TypedValue<RankedTensorType> inputValue,
                            RewriterBase &rewriter, Layout targetTensorLayout,
                            BufferType targetTensorBufferType,
                            TensorMemoryLayoutAttr targetTensorMemoryLayout,
                            ttcore::DataType targetTensorDataType,
                            llvm::StringRef locSuffix = "");

// Creates a ToLayoutOp that converts input to height-sharded L1 Tile layout
// with virtual grid [batchSize, 1]. Required by NLPConcatHeadsDecodeOp which
// expects each batch element assigned to its own row of cores.
ToLayoutOp createHeightShardedToLayout(mlir::Operation *op, mlir::Value input,
                                       RankedTensorType inputType,
                                       int64_t batchSize,
                                       RewriterBase &rewriter,
                                       mlir::Location loc);
} // namespace mlir::tt::ttnn::utils

#endif
