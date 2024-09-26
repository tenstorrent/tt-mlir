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

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.cpp.inc"

bool memoryConfigAttrEqual(const MemoryConfigAttr &lhs,
                           const MemoryConfigAttr &rhs) {
  bool isEqual = true;
  isEqual &= lhs.getTensorMemoryLayout().getValue() ==
             rhs.getTensorMemoryLayout().getValue();
  isEqual &= lhs.getBufferType().getValue() == rhs.getBufferType().getValue();
  return isEqual;
}

void TTNNDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.cpp.inc"
      >();
}
