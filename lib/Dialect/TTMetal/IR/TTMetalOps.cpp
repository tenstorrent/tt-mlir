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
  ::mlir::MemRefType outputTy = getOutput().getType();
  MemorySpaceAttr memSpaceAttr =
      mlir::cast<MemorySpaceAttr>(outputTy.getMemorySpace());
  if (not isDeviceMemorySpace(memSpaceAttr.getValue())) {
    return emitOpError("Output tensor must be in device memory space");
  }
  return success();
}

::mlir::LogicalResult EnqueueReadBufferOp::verify() {
  ::mlir::MemRefType outputTy = getOutput().getType();
  MemorySpaceAttr memSpaceAttr =
      mlir::dyn_cast_if_present<MemorySpaceAttr>(outputTy.getMemorySpace());
  if (memSpaceAttr && not isSystemMemorySpace(memSpaceAttr.getValue())) {
    return emitOpError("Output tensor must be in system memory space");
  }
  return success();
}

::mlir::LogicalResult CreateBufferOp::verify() {
  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto memref = getResult().getType();
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
    ::mlir::MemRefType outputTy = mlir::cast<MemRefType>(operand.getType());
    MemorySpaceAttr memSpaceAttr =
        mlir::cast<MemorySpaceAttr>(outputTy.getMemorySpace());
    if (not isDeviceMemorySpace(memSpaceAttr.getValue())) {
      return emitOpError("Input tensor must be in device memory space");
    }
  }
  return success();
}

} // namespace mlir::tt::ttmetal
