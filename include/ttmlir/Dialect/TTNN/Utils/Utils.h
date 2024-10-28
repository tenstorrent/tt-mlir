// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_UTILS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

namespace mlir::tt::ttnn::utils {

// Map TT::MemorySpace to TTNN::BufferType
//
mlir::tt::ttnn::BufferType
toTTNNBufferType(const mlir::tt::MemorySpace memorySpace);

// Map TT::TensorMemoryLayout to TTNN::TensorMemoryLayout
//
ttnn::TensorMemoryLayout
toTTNNTensorMemoryLayout(const tt::TensorMemoryLayout ttTensorMemoryLayout);

DataType getDataTypeFromMemRef(mlir::MemRefType memref);

Layout getLayoutFromMemRef(mlir::MemRefType memref);

mlir::Type createRowMajorTypeFromDtype(::mlir::MLIRContext *context,
                                       DataType dtype);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
