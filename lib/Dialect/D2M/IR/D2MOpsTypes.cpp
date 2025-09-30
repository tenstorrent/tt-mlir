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
