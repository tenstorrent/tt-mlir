// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.cpp.inc"

namespace mlir::tt::sfpi {

bool isSFPIVectorType(mlir::Type type, llvm::StringRef elementType) {
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vectorType) {
    return false;
  }

  // Check if it's a 4x8 vector (shape should be [4, 8])
  auto shape = vectorType.getShape();
  if (shape.size() != 2 || shape[0] != 4 || shape[1] != 8) {
    return false;
  }

  // Check element type
  auto elemType = vectorType.getElementType();
  if (elementType == "f32") {
    return elemType.isF32();
  } else if (elementType == "i32") {
    return elemType.isInteger(32); // Accept any 32-bit integer type
  } else if (elementType == "ui32") {
    return elemType.isInteger(32) && elemType.isUnsignedInteger();
  }

  return false;
}

bool isDstRegType(mlir::Type type) {
  // Implementation for SFPI destination register type checking
  // This would be implemented based on the actual register type definitions
  return false; // Placeholder
}

bool isLRegType(mlir::Type type) {
  // Implementation for SFPI local register type checking
  // This would be implemented based on the actual register type definitions
  return false; // Placeholder
}

} // namespace mlir::tt::sfpi
