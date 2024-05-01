// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/TTDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/TTOps.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt;

#include "ttmlir/Dialect/TT/TTOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/TT/TTOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// TT dialect.
//===----------------------------------------------------------------------===//

void TTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TT/TTOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/TT/TTOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
}
