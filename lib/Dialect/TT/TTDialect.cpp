// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/TTDialect.h"

#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/TTOps.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"

using namespace mlir;
using namespace mlir::tt;

#include "ttmlir/Dialect/TT/TTOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TT dialect.
//===----------------------------------------------------------------------===//

void TTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TT/TTOps.cpp.inc"
      >();
  registerTypes();
}
