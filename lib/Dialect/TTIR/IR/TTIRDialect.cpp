// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir;
using namespace mlir::tt::ttir;

#include "ttmlir/Dialect/TTIR/IR/TTIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TTIR dialect.
//===----------------------------------------------------------------------===//

void TTIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"
      >();
}
