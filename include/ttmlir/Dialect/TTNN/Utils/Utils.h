// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_UTILS_H

#include <llvm/Support/CommandLine.h>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::utils {

// Map tt::MemorySpace to ttnn::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::MemorySpace memorySpace);

// Map tt::TensorMemoryLayout to ttnn::TensorMemoryLayout
//
ttnn::TensorMemoryLayout
toTTNNTensorMemoryLayout(const tt::TensorMemoryLayout ttTensorMemoryLayout);

// Map ttnn::BufferType to tt::MemorySpace
//
mlir::tt::TensorMemoryLayout toTTTensorMemoryLayout(
    const ::mlir::tt::ttnn::TensorMemoryLayout ttnnTensorMemoryLayout);

// Map ttnn::BufferType to tt::MemorySpace
//
mlir::tt::MemorySpace
toTTMemorySpace(const mlir::tt::ttnn::BufferType bufferType);

// Get Layout from MemRefType
//
Layout getLayoutFromMemRef(mlir::MemRefType memref);

mlir::Type createRowMajorTypeFromDtype(::mlir::MLIRContext *context,
                                       DataType dtype);

// Helper method to create a RankedTensorType with the given encoding
RankedTensorType
createRankedTensorTypeWithEncoding(RankedTensorType tensorType,
                                   ttnn::TTNNLayoutAttr encoding);

// Return the L1 memory usage of the output tensor of the given op.
// Used within L1 interleaved policies.
//
uint64_t getOpOutputL1Usage(Operation *op, TTNNLayoutAttr opLayout,
                            DeviceAttr &deviceAttr);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
