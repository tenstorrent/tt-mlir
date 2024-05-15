// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt;

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"

MemorySpace LayoutAttr::getMemorySpace() const {
  return getMemref().getMemorySpace().template cast<mlir::tt::MemorySpaceAttr>().getValue();
}

void TTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"
      >();
}
