// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_UTILS_H

#include "mlir/IR/Value.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::utils {

// Attribute name for storing tensor L1 usage cap during optimizer passes.
// This attribute is set by DevicePassesWrapper and read by validation
// and analysis passes to avoid parameter threading through pass infrastructure.
inline constexpr llvm::StringLiteral g_TensorL1UsageCapAttrName =
    "ttnn.tensor_l1_usage_cap";

// Helper function to retrieve tensor L1 usage cap from module attribute.
// Returns the configured cap if found, otherwise returns the default value.
float getTensorL1UsageCap(Operation *op, float defaultValue = 0.95f);

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

// Helper method to get the buffer type from the tensor layout encoding.
BufferType getBufferTypeFromTensor(RankedTensorType tensorType);

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

// Helper method to create a NDShardSpecAttr if needed.
std::optional<NDShardSpecAttr>
createNDShardSpecIfNeeded(TTNNNDLayoutAttr layout);

bool isTTNNHoistGenericViaD2MOp(mlir::Operation *op);

// Returns all TTNN dialect registered operations.
std::set<mlir::StringRef> getAllTTNNDialectOps(MLIRContext *context);

// Check if operation's first result uses TTNN layout encoding.
bool producesTTNNLayoutEncoding(Operation *op);

// Check if operation's first result uses DRAM buffer layout.
bool producesDRAMLayout(Operation *op);

// Check if operation's first result uses L1 buffer layout.
bool producesL1Layout(Operation *op);

// Check if operation's first result uses system memory layout.
bool producesSystemMemoryLayout(Operation *op);

// Check if operation's first result uses tiled tensor layout.
bool producesTiledTensorLayout(Operation *op);

// Check if operation's first result uses sharded L1 layout.
bool producesShardedL1Layout(Operation *op);

// Check if operation's first operand uses DRAM buffer layout.
bool hasFirstOperandInDRAM(Operation *op);

mlir::RankedTensorType getTraceIdType(MLIRContext *ctx);

// Convert activation string to UnaryWithParamAttr.
// Returns nullptr if activation is not set or not recognized.
UnaryWithParamAttr getActivationAttr(MLIRContext *ctx,
                                     std::optional<StringRef> activation);

// Compute the bounding box grid dimensions from a layout's shard grid.
// Returns {gridX, gridY} representing the physical core grid extent.
std::pair<int64_t, int64_t> getPhysicalGridDimensions(TTNNLayoutAttr layout);

// Calculate optimal C_in_block for Conv3d based on channel count.
// Finds the largest block size in [32, 128] (step 32) that evenly divides
// in_channels. Returns 0 when no valid block size exists (TT-Metal will
// use the full channel count).
inline uint32_t calculateOptimalCInBlock(uint32_t in_channels) {
  for (uint32_t i = 128; i >= 32; i -= 32) {
    if (in_channels % i == 0) {
      return i;
    }
  }
  return 0;
}

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
