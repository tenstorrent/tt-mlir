// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::utils {

bool isTensorOnDevice(::mlir::RankedTensorType tensorType);

// Map ttcore::MemorySpace to ttnn::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::ttcore::MemorySpace memorySpace);

// Map ttnn::BufferType to ttcore::MemorySpace
//
mlir::tt::ttcore::MemorySpace
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

  static RankedTensorType create(RankedTensorType tensorType,
                                 mlir::tt::ttcore::GridAttr grid);

  static RankedTensorType create(RankedTensorType tensorType,
                                 mlir::tt::ttcore::DataType);

  static RankedTensorType create(RankedTensorType tensorType,
                                 ArrayRef<int64_t> tensorShape);
};

// Return the L1 memory usage of the output tensor of the given op.
// Used within L1 interleaved policies and temporarily within L1 Interleaved
// Fallback Analysis.
//
uint64_t getOpOutputL1Usage(TTNNLayoutAttr opLayout);

// Helper method to get the tensor layout attribute from the tensor value.
TTNNLayoutAttr getLayoutAttrFromTensor(RankedTensorType tensorType);

// Helper method to get the element type for the given tensor layout and data.
Type getElementType(MLIRContext *context, Layout tensorLayout,
                    mlir::tt::ttcore::DataType dataType);

// Helper method to get op location name if it exists. Else return empty string.
std::string getOpLocName(Operation *op);

// Save the IR to a file for debugging.
void irToFile(mlir::Operation *op, std::string filename);

// Convert a logical tensor shape to a tiled shape by rounding up the last two
// dims to tile size (32). E.g. (1, 2, 16, 16) -> (1, 2, 32, 32).
llvm::SmallVector<int64_t> getTilePaddedShape(llvm::ArrayRef<int64_t> shape);

// Extract input layouts from operation operands, skipping device type operands.
std::vector<TTNNLayoutAttr> extractInputLayouts(Operation *op);

// Helper method to create a ShardSpecAttr if needed.
std::optional<ShardSpecAttr>
createShardSpecIfNeeded(TTNNLayoutAttr layout,
                        mlir::tt::ttcore::GridAttr deviceGrid);

// Helper method to create a ShardSpecAttr if needed.
std::optional<ShardSpecAttr>
createShardSpecIfNeeded(TensorMemoryLayoutAttr tensorMemoryLayout,
                        ShapeAttr shardShape,
                        mlir::tt::ttcore::GridAttr shardGrid,
                        mlir::tt::ttcore::GridAttr deviceGrid);

bool isTTNNTraceFunc(func::FuncOp funcOp);
bool isTTNNHoistGenericViaD2MOp(mlir::Operation *op);

// Returns all TTNN dialect registered operations.
std::set<mlir::StringRef> getAllTTNNDialectOps(MLIRContext *context);

// Check if operation's first result uses TTNN layout encoding.
bool producesTTNNLayoutEncoding(Operation *op);

// Check if operation's first result uses DRAM buffer layout.
bool producesDRAMLayout(Operation *op);

// Check if operation's first result uses L1 buffer layout.
bool producesL1Layout(Operation *op);

// Check if operation's first result uses tiled tensor layout.
bool producesTiledTensorLayout(Operation *op);

// Check if operation's first operand uses DRAM buffer layout.
bool hasFirstOperandInDRAM(Operation *op);

mlir::RankedTensorType getTraceIdType(MLIRContext *ctx);

// Will return true if the operation has non-read memory effects on the provided
// Value.
bool operationHasNonReadMemoryEffectsOnValue(mlir::Value value,
                                             mlir::Operation *op);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
