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

inline bool changesDataType(mlir::tt::ttnn::TTNNLayoutAttr input,
                            mlir::tt::ttnn::TTNNLayoutAttr output) {
  return input.getScalarElementType() != output.getScalarElementType();
}

inline bool changesMemoryLayout(mlir::tt::ttnn::TTNNLayoutAttr input,
                                mlir::tt::ttnn::TTNNLayoutAttr output) {
  return input.getMemLayout() != output.getMemLayout();
}

inline bool changesBufferType(mlir::tt::ttnn::TTNNLayoutAttr input,
                              mlir::tt::ttnn::TTNNLayoutAttr output) {
  return input.getBufferType() != output.getBufferType();
}

inline bool changesTensorLayout(mlir::tt::ttnn::TTNNLayoutAttr input,
                                mlir::tt::ttnn::TTNNLayoutAttr output) {
  return input.getLayout() != output.getLayout();
}

#endif // TTMLIR_DIALECT_TTNN_IR_TTNNOPSATTRS_H
