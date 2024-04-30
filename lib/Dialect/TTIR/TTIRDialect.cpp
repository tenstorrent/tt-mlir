// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/TTIRDialect.h"
#include "ttmlir/Dialect/TT/TTDialect.h"

#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TTIR/TTIROps.h"

using namespace mlir;
using namespace mlir::tt::ttir;

#include "ttmlir/Dialect/TTIR/TTIROpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TTIR dialect.
//===----------------------------------------------------------------------===//

void TTIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTIR/TTIROps.cpp.inc"
      >();
}
