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

  // `deviceGrid` is required only when re-encoding to a sharded layout; pass
  // a null GridAttr (the default) for non-sharded targets.
  static RankedTensorType
  create(RankedTensorType tensorType, ttnn::BufferType bufferType,
         mlir::tt::ttcore::GridAttr deviceGrid = nullptr);

  static RankedTensorType
  create(RankedTensorType tensorType, ttnn::TensorMemoryLayout memoryLayout,
         mlir::tt::ttcore::GridAttr deviceGrid = nullptr);

  static RankedTensorType create(RankedTensorType tensorType,
                                 ttnn::Layout layout);

  // Re-encode `tensorType` with `gridShape` as the new shard grid. `deviceGrid`
  // is the physical worker grid used to derive the canonical CoreRangeSet for
  // sharded layouts. Pass the actual device's worker grid here, not a
  // synthesized GridAttr built from `gridShape`.
  static RankedTensorType create(RankedTensorType tensorType,
                                 ArrayRef<int64_t> gridShape,
                                 mlir::tt::ttcore::GridAttr deviceGrid);

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

// Return the per-core L1 memory usage of a layout.
// For sharded layouts, returns the shard size.
// For L1 interleaved, returns total size / numCores since the grid attribute
// is irrelevant for interleaved — data is distributed across all device cores.
uint64_t getPerCoreL1Usage(TTNNLayoutAttr layout, uint64_t numCores);

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
// Precondition: layout has a non-null CoreRangeSet (i.e. is sharded).
std::pair<int64_t, int64_t> getPhysicalGridDimensions(TTNNLayoutAttr layout);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
