// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include <llvm/Support/Casting.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace mlir::tt::ttnn::utils {
// Map TT::MemorySpace to TTNN::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::MemorySpace memorySpace) {
  switch (memorySpace) {
  case MemorySpace::System:
  case MemorySpace::SystemMMIO:
    return BufferType::SystemMemory;
  case MemorySpace::DeviceDRAM:
    return BufferType::DRAM;
  case MemorySpace::DeviceL1:
    return BufferType::L1;
  }

  llvm_unreachable("Unknown MemorySpace");
}

mlir::tt::MemorySpace
toTTMemorySpace(const mlir::tt::ttnn::BufferType bufferType) {
  switch (bufferType) {
  case ttnn::BufferType::SystemMemory:
    return MemorySpace::System;
  case ttnn::BufferType::DRAM:
    return MemorySpace::DeviceDRAM;
  case ttnn::BufferType::L1:
    return MemorySpace::DeviceL1;
  case ttnn::BufferType::L1Small:
    assert(false && "BufferType::L1Small not supported");
  case ttnn::BufferType::Trace:
    assert(false && "BufferType::Trace not supported");
  }

  llvm_unreachable("Unknown MemorySpace");
}

// Helper method to create a RankedTensorType with the given encoding.
RankedTensorType
createRankedTensorTypeWithEncoding(RankedTensorType tensorType,
                                   ttnn::TTNNLayoutAttr encoding) {
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

// Helper method to create a RankedTensorType with the given element type.
RankedTensorType
createRankedTensorTypeWithElementType(RankedTensorType tensorType,
                                      Type elementType) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding = oldEncoding.withElementType(
      tensorType.getContext(), elementType, tensorType.getShape());
  Type newElementType = elementType;
  if (TileType tileType = dyn_cast<TileType>(elementType)) {
    newElementType = tileType.getElementType();
  }
  return RankedTensorType::get(tensorType.getShape(), newElementType,
                               newEncoding);
}

// Helper method to create a RankedTensorType with the given buffer type.
RankedTensorType
createRankedTensorTypeWithBufferType(RankedTensorType tensorType,
                                     ttnn::BufferType bufferType) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding =
      oldEncoding.withBufferType(tensorType.getContext(), bufferType);
  return createRankedTensorTypeWithEncoding(tensorType, newEncoding);
}

// Helper method to create a RankedTensorType with the given memory layout.
RankedTensorType
createRankedTensorTypeWithMemoryLayout(RankedTensorType tensorType,
                                       ttnn::TensorMemoryLayout memoryLayout) {
  TTNNLayoutAttr oldEncoding = getLayoutAttrFromTensor(tensorType);
  TTNNLayoutAttr newEncoding =
      oldEncoding.withMemoryLayout(tensorType.getContext(), memoryLayout);
  return createRankedTensorTypeWithEncoding(tensorType, newEncoding);
}

// Return the L1 memory usage of the output tensor of the given op.
// Used within L1 interleaved policies.
//
uint64_t getOpOutputL1Usage(TTNNLayoutAttr opLayout) {
  // In case the opLayout is not in L1 memory space, L1 memory usage is 0.
  //
  if (opLayout.hasDRAMBufferType()) {
    return 0;
  }

  return opLayout.getShardSizeInBytes();
}

// Helper method to get the tensor layout attribute from the value.
TTNNLayoutAttr getLayoutAttrFromTensor(RankedTensorType tensorType) {
  return mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
}

// Helper method to get the element type for the given tensor layout and data.
Type getElementType(MLIRContext *context, Layout tensorLayout,
                    DataType dataType) {
  return tensorLayout == Layout::Tile
             ? TileType::get(context, {ttnn::TILE_HEIGHT, ttnn::TILE_WIDTH},
                             dataType)
             : mlir::tt::dataTypeToElementType(context, dataType);
}

// Save the IR to a file for debugging.
void irToFile(mlir::Operation *op, std::string filename) {
  OpPrintingFlags printFlags;
  printFlags = printFlags.enableDebugInfo();

  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);
  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
    return;
  }
  op->print(file, printFlags);
}

std::string getOpLocName(Operation *op) {
  if (NameLoc loc = llvm::dyn_cast<NameLoc>(op->getLoc())) {
    return loc.getName().str();
  }
  return "";
}

} // namespace mlir::tt::ttnn::utils
