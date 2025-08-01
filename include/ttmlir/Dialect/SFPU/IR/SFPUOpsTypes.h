// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPU_IR_SFPUOPSTYPES_H
#define TTMLIR_DIALECT_SFPU_IR_SFPUOPSTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::tt::sfpu {

// Type checking utilities for SFPU vector types
bool isSFPUVectorType(mlir::Type type, llvm::StringRef elementType);
bool isDstRegType(mlir::Type type);
bool isLRegType(mlir::Type type);

} // namespace mlir::tt::sfpu

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/SFPU/IR/SFPUOpsTypes.h.inc"

#endif