// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"

#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

using namespace mlir;
using namespace mlir::tt::ttkernel;

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TTKernel dialect.
//===----------------------------------------------------------------------===//

void TTKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.cpp.inc"
      >();
}
