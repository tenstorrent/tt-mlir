// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"

#include "ttmlir/Dialect/EmitPy/IR/EmitPy.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::emitpy;

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsTypes.cpp.inc"

void EmitPyDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOpsTypes.cpp.inc"
      >();
}

::mlir::LogicalResult mlir::tt::emitpy::OpaqueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef value) {
  if (value.empty()) {
    return emitError() << "expected non-empty string in !emitpy.opaque type";
  }
  return success();
}
