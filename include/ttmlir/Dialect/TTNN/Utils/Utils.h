// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::utils {

constexpr inline llvm::StringLiteral g_TTNNTraceAttrName = "ttnn.trace";

// Map tt::MemorySpace to ttnn::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::MemorySpace memorySpace);

// Map ttnn::BufferType to tt::MemorySpace
//
mlir::tt::MemorySpace
toTTMemorySpace(const mlir::tt::ttnn::BufferType bufferType);

struct RankedTensorTypeFactory {
  static RankedTensorType create(RankedTensorType tensorType,
                                 ttnn::TTNNLayoutAttr encoding);

  static RankedTensorType create(RankedTensorType tensorType,
                                 Type memrefElementType);

  static RankedTensorType create(RankedTensorType tensorType,
                                 ttnn::BufferType bufferType);

  static RankedTensorType create(RankedTensorType tensorType,
                                 ttnn::TensorMemoryLayout memoryLayout);

  static RankedTensorType create(RankedTensorType tensorType,
                                 ttnn::Layout layout);

  static RankedTensorType create(RankedTensorType tensorType, GridAttr grid);

  static RankedTensorType create(RankedTensorType tensorType, DataType);

  static RankedTensorType create(RankedTensorType tensorType,
                                 ArrayRef<int64_t> tensorShape);
};

// Return the L1 memory usage of the output tensor of the given op.
// Used within L1 interleaved policies.
//
uint64_t getOpOutputL1Usage(TTNNLayoutAttr opLayout);

// Helper method to get the tensor layout attribute from the tensor value.
TTNNLayoutAttr getLayoutAttrFromTensor(RankedTensorType tensorType);

// Helper method to get the element type for the given tensor layout and data.
Type getElementType(MLIRContext *context, Layout tensorLayout,
                    DataType dataType);

// Helper method to get op location name if it exists. Else return empty string.
std::string getOpLocName(Operation *op);

// Save the IR to a file for debugging.
void irToFile(mlir::Operation *op, std::string filename);

// Convert a logical tensor shape to a tiled shape by rounding up the last two
// dims to tile size (32). E.g. (1, 2, 16, 16) -> (1, 2, 32, 32).
llvm::SmallVector<int64_t> getTilePaddedShape(llvm::ArrayRef<int64_t> shape);

// Helper method to create a ShardSpecAttr if needed.
std::optional<ShardSpecAttr> createShardSpecIfNeeded(TTNNLayoutAttr layout,
                                                     GridAttr deviceGrid);

// Helper method to create a ShardSpecAttr if needed.
std::optional<ShardSpecAttr>
createShardSpecIfNeeded(TensorMemoryLayoutAttr tensorMemoryLayout,
                        ShapeAttr shardShape, GridAttr shardGrid,
                        GridAttr deviceGrid);

bool isTTNNTraceFunc(func::FuncOp funcOp);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
