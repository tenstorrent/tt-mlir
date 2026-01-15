// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/Debug/IR/Debug.h"
#include "ttmlir/Dialect/Debug/IR/DebugOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttmlir/Dialect/Debug/IR/DebugOpsDialect.cpp.inc"

namespace mlir::tt::debug {

//===----------------------------------------------------------------------===//
// Debug dialect.
//===----------------------------------------------------------------------===//

void DebugDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/Debug/IR/DebugOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Dialect parser/printer methods
//===----------------------------------------------------------------------===//

mlir::Type DebugDialect::parseType(mlir::DialectAsmParser &parser) const {
  // For now, delegate to generated parser
  StringRef typeTag;
  if (parser.parseKeyword(&typeTag)) {
    return {};
  }

  // Handle any Debug-specific types here
  // Currently we don't have any custom types
  parser.emitError(parser.getNameLoc(), "unknown Debug dialect type: ")
      << typeTag;
  return {};
}

void DebugDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &os) const {
  // Handle any Debug-specific types here
  // Currently we don't have any custom types
  os << "<<unknown Debug type>>";
}

mlir::Attribute DebugDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                             mlir::Type type) const {
  // For now, delegate to generated parser
  mlir::StringRef attrName;
  if (parser.parseKeyword(&attrName)) {
    return {};
  }

  // Handle any Debug-specific attributes here
  // Currently we don't have any custom attributes
  parser.emitError(parser.getNameLoc(), "unknown Debug dialect attribute: ")
      << attrName;
  return {};
}

void DebugDialect::printAttribute(mlir::Attribute attr,
                                  mlir::DialectAsmPrinter &printer) const {
  // Handle any Debug-specific attributes here
  // Currently we don't have any custom attributes
  printer << "<unknown debug attribute>";
}

mlir::Operation *DebugDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  // For now, we cannot materialize any constants in the Debug dialect
  return nullptr;
}

} // namespace mlir::tt::debug
