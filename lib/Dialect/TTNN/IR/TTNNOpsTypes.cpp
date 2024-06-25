// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttnn;

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.cpp.inc"

void TTNNDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.cpp.inc"
      >();
}
