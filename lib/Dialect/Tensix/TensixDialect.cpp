// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/Tensix/TensixDialect.h"

#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/Tensix/TensixOps.h"

using namespace mlir;
using namespace mlir::tt::tensix;

#include "ttmlir/Dialect/Tensix/TensixOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tensix dialect.
//===----------------------------------------------------------------------===//

void TensixDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/Tensix/TensixOps.cpp.inc"
      >();
}
