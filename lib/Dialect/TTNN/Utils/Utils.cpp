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

} // namespace mlir::tt::ttnn::utils
