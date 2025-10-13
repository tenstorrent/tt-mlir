// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SFPI/IR/SFPI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/SFPI/IR/SFPIOps.h"
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttmlir/Dialect/SFPI/IR/SFPIOpsDialect.cpp.inc"
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsAttrs.cpp.inc"

namespace mlir::tt::sfpi {

void SFPIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/SFPI/IR/SFPIOps.cpp.inc"
      >();
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsAttrs.cpp.inc"
      >();
  registerTypes();
}

void SFPIDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/SFPI/IR/SFPIOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Dialect parser/printer methods
//===----------------------------------------------------------------------===//
// Generated attribute parsers/printers are included automatically

mlir::Type SFPIDialect::parseType(mlir::DialectAsmParser &parser) const {
  // For now, delegate to generated parser
  StringRef typeTag;
  if (parser.parseKeyword(&typeTag)) {
    return {};
  }

  // Handle any SFPI-specific types here
  // Currently we don't have any custom types
  parser.emitError(parser.getNameLoc(), "unknown SFPI dialect type: ")
      << typeTag;
  return {};
}

void SFPIDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &os) const {
  // Handle any SFPI-specific types here
  // Currently we don't have any custom types
  os << "<<unknown SFPI type>>";
}

} // namespace mlir::tt::sfpi
