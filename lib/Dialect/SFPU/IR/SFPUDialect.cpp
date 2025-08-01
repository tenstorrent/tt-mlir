// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/SFPU/IR/SFPU.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/InitAllDialects.h"
#include "ttmlir/Dialect/SFPU/IR/SFPUOps.h"
#include "ttmlir/Dialect/SFPU/IR/SFPUOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttmlir/Dialect/SFPU/IR/SFPUOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttmlir/Dialect/SFPU/IR/SFPUOpsAttrDefs.cpp.inc"

namespace mlir::tt::sfpu {

void SFPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttmlir/Dialect/SFPU/IR/SFPUOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttmlir/Dialect/SFPU/IR/SFPUOpsAttrDefs.cpp.inc"
      >();
  registerTypes();
}

void SFPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/SFPU/IR/SFPUOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::tt::sfpu