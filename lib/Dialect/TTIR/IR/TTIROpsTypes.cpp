// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttir;

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypeDefs.cpp.inc"

void TTIRDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypeDefs.cpp.inc"
      >();
}
