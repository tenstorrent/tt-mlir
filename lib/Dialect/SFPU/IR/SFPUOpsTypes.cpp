// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SFPU/IR/SFPUOpsTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/SFPU/IR/SFPUOpsTypes.cpp.inc"

namespace mlir::tt::sfpu {

bool isSFPUVectorType(mlir::Type type, llvm::StringRef elementType) {
  auto vectorType = llvm::dyn_cast<mlir::VectorType>(type);
  if (!vectorType) {
    return false;
  }
  
  // Check if it's a 64-element vector
  if (vectorType.getNumElements() != 64) {
    return false;
  }
  
  // Check element type
  auto elemType = vectorType.getElementType();
  if (elementType == "f32") {
    return elemType.isF32();
  } else if (elementType == "i32") {
    return elemType.isInteger(32) && elemType.isSignedInteger();
  } else if (elementType == "ui32") {
    return elemType.isInteger(32) && elemType.isUnsignedInteger();
  }
  
  return false;
}

bool isDstRegType(mlir::Type type) {
  // Implementation for SFPU destination register type checking
  // This would be implemented based on the actual register type definitions
  return false; // Placeholder
}

bool isLRegType(mlir::Type type) {
  // Implementation for SFPU local register type checking
  // This would be implemented based on the actual register type definitions  
  return false; // Placeholder
}

} // namespace mlir::tt::sfpu