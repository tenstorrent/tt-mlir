// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/TTOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt;

#include "ttmlir/Dialect/TT/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/TTOpsTypes.cpp.inc"

MemorySpace LayoutAttr::getMemorySpace() {
  return getMemref().getMemorySpace().template cast<mlir::tt::MemorySpaceAttr>().getValue();
}

void TTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/TTOpsTypes.cpp.inc"
      >();
}
