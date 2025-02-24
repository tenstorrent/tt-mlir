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

// Map TT::TensorMemoryLayout to TTNN::TensorMemoryLayout
//
mlir::tt::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    const ::mlir::tt::TensorMemoryLayout ttTensorMemoryLayout) {

  switch (ttTensorMemoryLayout) {
  case ::mlir::tt::TensorMemoryLayout::HeightSharded:
    return ttnn::TensorMemoryLayout::HeightSharded;
  case ::mlir::tt::TensorMemoryLayout::Interleaved:
    return ttnn::TensorMemoryLayout::Interleaved;
  case ::mlir::tt::TensorMemoryLayout::WidthSharded:
    return ttnn::TensorMemoryLayout::WidthSharded;
  case ::mlir::tt::TensorMemoryLayout::BlockSharded:
    return ttnn::TensorMemoryLayout::BlockSharded;
  case ::mlir::tt::TensorMemoryLayout::SingleBank:
    return ttnn::TensorMemoryLayout::SingleBank;
  default:
    llvm_unreachable("Unknown TensorMemoryLayout");
  }
}

mlir::tt::TensorMemoryLayout toTTTensorMemoryLayout(
    const ::mlir::tt::ttnn::TensorMemoryLayout ttnnTensorMemoryLayout) {

  switch (ttnnTensorMemoryLayout) {
  case ttnn::TensorMemoryLayout::HeightSharded:
    return ::mlir::tt::TensorMemoryLayout::HeightSharded;
  case ttnn::TensorMemoryLayout::Interleaved:
    return ::mlir::tt::TensorMemoryLayout::Interleaved;
  case ttnn::TensorMemoryLayout::WidthSharded:
    return ::mlir::tt::TensorMemoryLayout::WidthSharded;
  case ttnn::TensorMemoryLayout::BlockSharded:
    return ::mlir::tt::TensorMemoryLayout::BlockSharded;
  case ttnn::TensorMemoryLayout::SingleBank:
    return ::mlir::tt::TensorMemoryLayout::SingleBank;
  }

  llvm_unreachable("Unknown TensorMemoryLayout");
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

Type dataTypeToElementType(::mlir::MLIRContext *context, DataType dtype) {
  switch (dtype) {
  case DataType::Float32:
    return FloatType::getF32(context);
  case DataType::Float16:
    return FloatType::getF16(context);
  case DataType::BFloat16:
    return FloatType::getBF16(context);
  case DataType::BFP_Float8:
    return FloatType::getF16(context);
  case DataType::BFP_BFloat8:
    return FloatType::getBF16(context);
  case DataType::BFP_Float4:
    return FloatType::getF16(context);
  case DataType::BFP_BFloat4:
    return FloatType::getBF16(context);
  case DataType::BFP_Float2:
    return FloatType::getF16(context);
  case DataType::BFP_BFloat2:
    return FloatType::getBF16(context);
  case DataType::UInt32:
    return IntegerType::get(context, 32,
                            IntegerType::SignednessSemantics::Unsigned);
  case DataType::UInt16:
    return IntegerType::get(context, 16,
                            IntegerType::SignednessSemantics::Unsigned);
  case DataType::UInt8:
    return IntegerType::get(context, 8,
                            IntegerType::SignednessSemantics::Unsigned);
  }
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
             : ttnn::utils::dataTypeToElementType(context, dataType);
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
