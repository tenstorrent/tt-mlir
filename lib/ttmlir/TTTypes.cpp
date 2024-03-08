// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/TTTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/TTDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt;

#define GET_TYPEDEF_CLASSES
#include "ttmlir/TTOpsTypes.cpp.inc"

void TTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/TTOpsTypes.cpp.inc"
      >();
}
