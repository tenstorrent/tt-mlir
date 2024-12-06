// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.h.inc"

namespace mlir::tt::ttnn {

inline bool isSystemBufferType(BufferType bufferType) {
  return bufferType == BufferType::SystemMemory;
}

inline bool isDeviceBufferType(BufferType bufferType) {
  return bufferType == BufferType::L1 || bufferType == BufferType::DRAM ||
         bufferType == BufferType::L1Small;
}

inline bool isL1BufferType(BufferType bufferType) {
  return bufferType == BufferType::L1;
}

inline bool isDRAMBufferType(BufferType bufferType) {
  return bufferType == BufferType::DRAM;
}

inline bool isShardedMemoryLayout(TensorMemoryLayout layout) {
  return layout == TensorMemoryLayout::HeightSharded ||
         layout == TensorMemoryLayout::WidthSharded ||
         layout == TensorMemoryLayout::BlockSharded;
}

} // namespace mlir::tt::ttnn

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrDefs.h.inc"

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H
