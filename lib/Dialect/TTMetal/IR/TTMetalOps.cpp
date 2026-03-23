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
  ttcore::MemorySpaceAttr memSpaceAttr =
      mlir::cast<ttcore::MemorySpaceAttr>(outputTy.getMemorySpace());
  if (not isDeviceMemorySpace(memSpaceAttr.getValue())) {
    return emitOpError("Output tensor must be in device memory space");
  }
  return success();
}

::mlir::LogicalResult EnqueueReadBufferOp::verify() {
  ::mlir::MemRefType outputTy = getOutput().getType();
  ttcore::MemorySpaceAttr memSpaceAttr =
      mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
          outputTy.getMemorySpace());
  if (memSpaceAttr && not isSystemMemorySpace(memSpaceAttr.getValue())) {
    return emitOpError("Output tensor must be in system memory space");
  }
  return success();
}

::mlir::LogicalResult EnqueueProgramOp::verify() {
  for (auto operand : getOperands()) {
    if (auto memRefType = mlir::dyn_cast<MemRefType>(operand.getType());
        memRefType) {
      ttcore::MemorySpaceAttr memSpaceAttr =
          mlir::cast<ttcore::MemorySpaceAttr>(memRefType.getMemorySpace());
      if (not isDeviceMemorySpace(memSpaceAttr.getValue())) {
        return emitOpError("Operand tensor to EnqueueProgramOp must be in "
                           "device memory space");
      }
    } else if (auto globalSemaphoreType =
                   mlir::dyn_cast<ttmetal::GlobalSemaphoreType>(
                       operand.getType());
               globalSemaphoreType) {
      continue;
    }
  }
  return success();
}

} // namespace mlir::tt::ttmetal
