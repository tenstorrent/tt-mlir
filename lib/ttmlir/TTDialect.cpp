// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/TTDialect.h"
#include "ttmlir/TTOps.h"
#include "ttmlir/TTTypes.h"

using namespace mlir;
using namespace mlir::tt;

#include "ttmlir/TTOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TT dialect.
//===----------------------------------------------------------------------===//

void TTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/TTOps.cpp.inc"
      >();
  registerTypes();
}
