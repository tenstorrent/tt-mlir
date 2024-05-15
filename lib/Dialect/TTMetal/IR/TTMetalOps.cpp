// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.cpp.inc"

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
  if (not inputLayout.isSystemMemorySpace()) {
    return emitOpError("Input tensor must be in system memory space");
  }
  if (not outputLayout.isDeviceMemorySpace()) {
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
  if (not outputLayout.isSystemMemorySpace()) {
    return emitOpError("Output tensor must be in system memory space");
  }
  if (not inputLayout.isDeviceMemorySpace()) {
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

  if (isSystemMemorySpace(getMemorySpace()) and getAddress() != 0) {
    return emitOpError("Allocating from system memory space must have address "
                       "set to 0, implicitly allocated by the runtime");
  }

  if (isDeviceMemorySpace(memspace) and getAddress() == 0) {
    return emitOpError(
        "Allocating from a device memory space must have address "
        "set to a non-zero value, device addresses are statically allocated");
  }

  return success();
}

::mlir::LogicalResult mlir::tt::ttmetal::DispatchOp::verify() {
  // Assert inputs/outputs device memspace
  for (auto operand : getOperands()) {
    auto layout = operand.getType()
                      .cast<mlir::RankedTensorType>()
                      .getEncoding()
                      .template dyn_cast_or_null<mlir::tt::LayoutAttr>();
    if (not layout) {
      return emitOpError("Input tensor missing layout attribute");
    }
    if (not layout.isDeviceMemorySpace()) {
      return emitOpError("Input tensor must be in device memory space");
    }
  }

  // Assert block inputs are CBs
  for (auto &region : getRegions()) {
    for (auto arg : region.getArguments()) {
      if (not arg.getType().isa<mlir::tt::ttmetal::CBType>()) {
        return emitOpError("Block inputs must be CBType");
      }
    }
  }
  return success();
}
