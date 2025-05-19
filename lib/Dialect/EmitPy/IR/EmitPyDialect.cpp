// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/InitAllDialects.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tt::emitpy;

//===----------------------------------------------------------------------===//
// EmitPy dialect.
//===----------------------------------------------------------------------===//

void EmitPyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.cpp.inc"
      >();
}
