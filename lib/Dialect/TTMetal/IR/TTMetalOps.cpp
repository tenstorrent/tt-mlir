// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.cpp.inc"

namespace mlir::tt::ttmetal {

::mlir::LogicalResult EnqueueWriteBufferOp::verify() {
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto outputLayout =
      mlir::dyn_cast_or_null<mlir::tt::MetalLayoutAttr>(outputTy.getEncoding());
  if (not outputLayout) {
    return emitOpError("Input tensor missing layout attribute");
  }
  if (not outputLayout.isDeviceMemorySpace()) {
    return emitOpError("Output tensor must be in device memory space");
  }
  return success();
}

::mlir::LogicalResult HostReadOp::verify() {
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto outputLayout =
      mlir::dyn_cast_or_null<mlir::tt::MetalLayoutAttr>(outputTy.getEncoding());
  if (not outputLayout) {
    return emitOpError("Input tensor missing layout attribute");
  }
  if (not outputLayout.isSystemMemorySpace()) {
    return emitOpError("Output tensor must be in system memory space");
  }
  return success();
}

::mlir::LogicalResult AllocOp::verify() {
  auto layout = mlir::dyn_cast_or_null<mlir::tt::MetalLayoutAttr>(
      getResult().getType().getEncoding());
  if (not layout) {
    return emitOpError("Result type missing layout attribute");
  }

  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto memref = layout.getMemref();
  auto memspace =
      mlir::cast<mlir::tt::MemorySpaceAttr>(memref.getMemorySpace()).getValue();
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

::mlir::LogicalResult EnqueueProgramOp::verify() {
  // Assert inputs/outputs device memspace
  for (auto operand : getOperands()) {
    auto layout = mlir::dyn_cast_or_null<mlir::tt::MetalLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(operand.getType()).getEncoding());
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
      if (not mlir::isa<ttkernel::CBType>(arg.getType())) {
        return emitOpError("Block inputs must be CBType");
      }
    }
  }
  return success();
}

} // namespace mlir::tt::ttmetal
