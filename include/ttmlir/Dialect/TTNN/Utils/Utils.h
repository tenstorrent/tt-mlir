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
TensorMemoryLayout
toTTNNTensorMemoryLayout(const TensorMemoryLayout TensorMemoryLayout);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_UTILS_H
