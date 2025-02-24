// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_UTILS_H

#include <llvm/Support/CommandLine.h>
#include <mlir/IR/Value.h>

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::utils {

// Map tt::MemorySpace to ttnn::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::MemorySpace memorySpace);

// Map ttnn::BufferType to tt::MemorySpace
//
mlir::tt::MemorySpace
toTTMemorySpace(const mlir::tt::ttnn::BufferType bufferType);

// Helper method to create a RankedTensorType with the given encoding.
RankedTensorType
createRankedTensorTypeWithEncoding(RankedTensorType tensorType,
                                   ttnn::TTNNLayoutAttr encoding);

// Helper method to create a RankedTensorType with the given element type.
RankedTensorType
createRankedTensorTypeWithElementType(RankedTensorType tensorType,
                                      Type elementType);

// Helper method to create a RankedTensorType with the given buffer type.
RankedTensorType
createRankedTensorTypeWithBufferType(RankedTensorType tensorType,
                                     ttnn::BufferType bufferType);

// Helper method to create a RankedTensorType with the given memory layout.
RankedTensorType
createRankedTensorTypeWithMemoryLayout(RankedTensorType tensorType,
                                       ttnn::TensorMemoryLayout memoryLayout);

// Return the L1 memory usage of the output tensor of the given op.
// Used within L1 interleaved policies.
//
uint64_t getOpOutputL1Usage(TTNNLayoutAttr opLayout);

// Helper method to get the tensor layout attribute from the tensor value.
TTNNLayoutAttr getLayoutAttrFromTensor(RankedTensorType tensorType);

// Helper method to get the element type for the given tensor layout and data.
Type getElementType(MLIRContext *context, Layout tensorLayout,
                    DataType dataType);

// Helper method to get op location name if it exists. Else return empty string.
std::string getOpLocName(Operation *op);

// Save the IR to a file for debugging.
void irToFile(mlir::Operation *op, std::string filename);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
