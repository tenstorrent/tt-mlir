// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

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
  case ::mlir::tt::TensorMemoryLayout::None:
    return ttnn::TensorMemoryLayout::None;
  }

  llvm_unreachable("Unknown TensorMemoryLayout");
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
  case ttnn::TensorMemoryLayout::None:
    return ::mlir::tt::TensorMemoryLayout::None;
  }
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

DataType getDataTypeFromMemRef(mlir::MemRefType memref) {
  Type elementType = memref.getElementType();
  DataType dtype = DataType::Float32;
  if (llvm::isa<TileType>(elementType)) {
    auto tileType = mlir::cast<TileType>(elementType);
    dtype = tileType.getDataType();
  } else {
    dtype = elementTypeToDataType(elementType);
  }
  return dtype;
}

Layout getLayoutFromMemRef(mlir::MemRefType memref) {
  ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
  Type elementType = memref.getElementType();
  if (llvm::isa<TileType>(elementType)) {
    ttnnLayoutEnum = ttnn::Layout::Tile;
  } else {
    ttnnLayoutEnum = ttnn::Layout::RowMajor;
  }
  return ttnnLayoutEnum;
}

Type createRowMajorTypeFromDtype(::mlir::MLIRContext *context, DataType dtype) {
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
    return IntegerType::get(context, 32);
  case DataType::UInt16:
    return IntegerType::get(context, 16);
  case DataType::UInt8:
    return IntegerType::get(context, 8);
  }
}

// Helper method to create a RankedTensorType with the given encoding
RankedTensorType
createRankedTensorTypeWithEncoding(RankedTensorType tensorType,
                                   ttnn::TTNNLayoutAttr encoding) {
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

} // namespace mlir::tt::ttnn::utils
