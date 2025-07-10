// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::emitpy;

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.cpp.inc"

void EmitPyDialect::registerAttributes() {
  // NOLINTNEXTLINE
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/EmitPy/IR/EmitPyAttrs.cpp.inc"
      >();
}
