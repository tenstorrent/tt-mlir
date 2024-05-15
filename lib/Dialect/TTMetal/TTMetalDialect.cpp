// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/TTMetalDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/TTMetal/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/TTMetalOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::ttmetal;

namespace mlir::tt::ttmetal {
static void printDimensionList(::mlir::AsmPrinter &printer,
                               ::llvm::ArrayRef<int64_t> shape) {
  printer.printDimensionList(shape);
}

static ::mlir::ParseResult
parseDimensionList(::mlir::AsmParser &odsParser,
                   ::llvm::SmallVector<int64_t> &dimensions) {
  return odsParser.parseDimensionList(dimensions, false, false);
}
} // namespace mlir::tt::ttmetal

#include "ttmlir/Dialect/TTMetal/TTMetalOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/TTMetalOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TTMetal dialect.
//===----------------------------------------------------------------------===//

void TTMetalDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTMetal/TTMetalOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TTMetal/TTMetalOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
}
