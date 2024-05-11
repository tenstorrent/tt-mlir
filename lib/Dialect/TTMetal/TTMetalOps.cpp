// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/TTMetalDialect.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTMetal/TTMetalOps.cpp.inc"

::mlir::LogicalResult mlir::tt::ttmetal::AllocOp::verify() {
  auto layout = getResult()
                    .getType()
                    .getEncoding()
                    .template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (not layout) {
    return emitOpError("Result type missing layout attribute");
  }

  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto memref = layout.getMemref();
  auto memspace = memref.getMemorySpace()
                      .template cast<mlir::tt::MemorySpaceAttr>()
                      .getValue();
  if (memspace != getMemorySpace()) {
    return emitOpError(
        "Input tensor layout memory space must match alloc memory space");
  }

  if (getMemorySpace() == mlir::tt::MemorySpace::System and getAddress() != 0) {
    return emitOpError("Allocating from system memory space must have address "
                       "set to 0, implicitly allocated by the runtime");
  }

  bool isDeviceMemorySpace = memspace == mlir::tt::MemorySpace::DRAM or
                             memspace == mlir::tt::MemorySpace::L1;
  if (isDeviceMemorySpace and getAddress() == 0) {
    return emitOpError(
        "Allocating from a device memory space must have address "
        "set to a non-zero value, device addresses are statically allocated");
  }

  return success();
}
