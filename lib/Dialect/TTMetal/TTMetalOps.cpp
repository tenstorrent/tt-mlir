// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/TTMetalDialect.h"
#include "ttmlir/Dialect/TTMetal/TTMetalOpsTypes.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTMetal/TTMetalOps.cpp.inc"

::mlir::LogicalResult mlir::tt::ttmetal::HostWriteOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto inputLayout =
      inputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (not inputLayout) {
    return emitOpError("Input tensor missing layout attribute");
  }
  auto outputLayout =
      outputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (not outputLayout) {
    return emitOpError("Input tensor missing layout attribute");
  }
  auto inputMemorySpace = inputLayout.getMemorySpace();
  auto outputMemorySpace = outputLayout.getMemorySpace();
  if (inputMemorySpace != mlir::tt::MemorySpace::System) {
    return emitOpError("Input tensor must be in system memory space");
  }
  if (outputMemorySpace != mlir::tt::MemorySpace::DRAM and
      outputMemorySpace != mlir::tt::MemorySpace::L1) {
    return emitOpError("Output tensor must be in device memory space");
  }
  return success();
}

::mlir::LogicalResult mlir::tt::ttmetal::HostReadOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto inputLayout =
      inputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (not inputLayout) {
    return emitOpError("Input tensor missing layout attribute");
  }
  auto outputLayout =
      outputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (not outputLayout) {
    return emitOpError("Input tensor missing layout attribute");
  }
  auto inputMemorySpace = inputLayout.getMemorySpace();
  auto outputMemorySpace = outputLayout.getMemorySpace();
  if (outputMemorySpace != mlir::tt::MemorySpace::System) {
    return emitOpError("Output tensor must be in system memory space");
  }
  if (inputMemorySpace != mlir::tt::MemorySpace::DRAM and
      inputMemorySpace != mlir::tt::MemorySpace::L1) {
    return emitOpError("Input tensor must be in device memory space");
  }
  return success();
}

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

::mlir::LogicalResult mlir::tt::ttmetal::DispatchOp::verify() {
  // Assert inputs/outputs device memspace
  // Assert block inputs are CBs
  return success();
}
