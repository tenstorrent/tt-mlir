// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/Tensix/IR/Tensix.h"

#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/Tensix/IR/TensixOps.h"

using namespace mlir;
using namespace mlir::tt::tensix;

#include "ttmlir/Dialect/Tensix/IR/TensixOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tensix dialect.
//===----------------------------------------------------------------------===//

void TensixDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/Tensix/IR/TensixOps.cpp.inc"
      >();
}
