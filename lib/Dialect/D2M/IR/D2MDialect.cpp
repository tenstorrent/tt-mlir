// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Ensure enum helpers (FieldParser, etc.) are visible before attrs
// The declarations live in D2MOps.h via D2MOpsEnums.h.inc; only include cpp
// here.
#include "ttmlir/Dialect/D2M/IR/D2MOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

#include "ttmlir/Dialect/D2M/IR/D2MOpsDialect.cpp.inc"

void D2MDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypeDefs.h.inc"
      >();
}

void D2MDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/D2M/IR/D2MOpsAttrs.cpp.inc"
      >();
  registerTypes();
}
