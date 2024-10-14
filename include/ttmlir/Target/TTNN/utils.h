// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTNN_UTILS_H
#define TTMLIR_TARGET_TTNN_UTILS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Target/Common/types_generated.h"
#include <llvm/Support/ErrorHandling.h>

namespace tt::mlir::ttnn::utils {

::tt::target::TensorMemoryLayout
toTargetTensorMemoryLayout(::mlir::tt::TensorMemoryLayout tensorMemoryLayout) {

  switch (tensorMemoryLayout) {
  case ::mlir::tt::TensorMemoryLayout::Interleaved:
    return ::tt::target::TensorMemoryLayout::Interleaved;
  case ::mlir::tt::TensorMemoryLayout::SingleBank:
    return ::tt::target::TensorMemoryLayout::SingleBank;
  case ::mlir::tt::TensorMemoryLayout::HeightSharded:
    return ::tt::target::TensorMemoryLayout::HeightSharded;
  case ::mlir::tt::TensorMemoryLayout::WidthSharded:
    return ::tt::target::TensorMemoryLayout::WidthSharded;
  case ::mlir::tt::TensorMemoryLayout::BlockSharded:
    return ::tt::target::TensorMemoryLayout::BlockSharded;
  case ::mlir::tt::TensorMemoryLayout::None:
    llvm_unreachable("Unsupported TensorMemoryLayout");
  }
}

::tt::target::BufferType
toTargetBufferType(::mlir::tt::ttnn::BufferType bufferType) {

  switch (bufferType) {
  case ::mlir::tt::ttnn::BufferType::DRAM:
    return ::tt::target::BufferType::DRAM;
  case ::mlir::tt::ttnn::BufferType::L1:
    return ::tt::target::BufferType::L1;
  case ::mlir::tt::ttnn::BufferType::SystemMemory:
    return ::tt::target::BufferType::SystemMemory;
  case ::mlir::tt::ttnn::BufferType::L1Small:
    return ::tt::target::BufferType::L1Small;
  case ::mlir::tt::ttnn::BufferType::Trace:
    return ::tt::target::BufferType::Trace;
  }

  llvm_unreachable("Unsupported BufferType");
}

::tt::target::TensorLayout
toTargetTensorLayout(::mlir::tt::ttnn::Layout layout) {
  switch (layout) {
  case ::mlir::tt::ttnn::Layout::RowMajor:
    return ::tt::target::TensorLayout::RowMajor;
  case ::mlir::tt::ttnn::Layout::Tile:
    return ::tt::target::TensorLayout::Tile;
  case ::mlir::tt::ttnn::Layout::Invalid:
    llvm_unreachable("Unsupported Layout");
  }

  llvm_unreachable("Unsupported Layout");
}

::tt::target::DataType toTargetDataType(::mlir::tt::DataType dataType) {
  switch (dataType) {
  case ::mlir::tt::DataType::BFloat16:
    return ::tt::target::DataType::BFloat16;
  case ::mlir::tt::DataType::Float32:
    return ::tt::target::DataType::Float32;
  case ::mlir::tt::DataType::UInt32:
    return ::tt::target::DataType::UInt32;
  case ::mlir::tt::DataType::BFP_BFloat8:
    return ::tt::target::DataType::BFP_BFloat8;
  case ::mlir::tt::DataType::BFP_BFloat4:
    return ::tt::target::DataType::BFP_BFloat4;
  case ::mlir::tt::DataType::UInt8:
    return ::tt::target::DataType::UInt8;
  case ::mlir::tt::DataType::UInt16:
    return ::tt::target::DataType::UInt16;
  case ::mlir::tt::DataType::Float16:
  case ::mlir::tt::DataType::BFP_Float2:
  case ::mlir::tt::DataType::BFP_Float4:
  case ::mlir::tt::DataType::BFP_Float8:
  case ::mlir::tt::DataType::BFP_BFloat2:
    llvm_unreachable("Unsupported DataType");
  }

  llvm_unreachable("Unsupported DataType");
}

} // namespace tt::mlir::ttnn::utils

#endif // TTMLIR_TARGET_TTNN_UTILS_H
