// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttmetal;

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalAttrInterfaces.cpp.inc"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp.inc"

void TTMetalDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.cpp.inc"
      >();
}
