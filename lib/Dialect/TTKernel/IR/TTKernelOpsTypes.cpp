// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::ttkernel;

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.cpp.inc"

void TTKernelDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.cpp.inc"
      >();
}
