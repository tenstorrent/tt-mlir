// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPI_IR_SFPIOPSTYPES_H
#define TTMLIR_DIALECT_SFPI_IR_SFPIOPSTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::tt::sfpi {

// Type checking utilities for SFPI vector types
bool isSFPIVectorType(mlir::Type type, llvm::StringRef elementType);
bool isDstRegType(mlir::Type type);
bool isLRegType(mlir::Type type);

} // namespace mlir::tt::sfpi

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.h.inc"

#endif