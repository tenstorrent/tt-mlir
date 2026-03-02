// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt::d2m;

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypeDefs.cpp.inc"

void D2MDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypeDefs.cpp.inc"
      >();
}

mlir::LogicalResult mlir::tt::d2m::ScalarType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type underlying) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(underlying)) {
    unsigned width = intTy.getWidth();
    if (width == 1 || width == 8 || width == 16 || width == 32) {
      return success();
    }
    return emitError() << "expected integer width of 1, 8, 16, or 32, got "
                       << width;
  }
  if (mlir::isa<mlir::Float32Type, mlir::Float16Type, mlir::BFloat16Type>(
          underlying)) {
    return success();
  }
  return emitError() << "expected bool (i1), integer (8/16/32-bit), or float "
                        "(f32/f16/bf16) type, got "
                     << underlying;
}
