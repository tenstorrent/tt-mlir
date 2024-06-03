// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
static void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

static ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}
} // namespace mlir::tt::ttnn

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TTNN dialect.
//===----------------------------------------------------------------------===//

void TTNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
}
