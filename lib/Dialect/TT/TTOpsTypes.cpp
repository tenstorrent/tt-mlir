// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/TTOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt;

void printGridDimension(::mlir::AsmPrinter &printer,
                        ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

::mlir::ParseResult
parseGridDimension(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}

#include "ttmlir/Dialect/TT/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/TTOpsTypes.cpp.inc"

void TTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/TTOpsTypes.cpp.inc"
      >();
}
