// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.cpp.inc"

namespace mlir::tt::ttnn {

::mlir::LogicalResult mlir::tt::ttnn::ToMemoryConfigOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto inputLayout =
      inputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  auto outputLayout =
      outputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (not inputLayout) {
    return emitOpError("Input tensor type missing layout attribute");
  }
  if (not outputLayout) {
    return emitOpError("Output tensor type missing layout attribute");
  }
  return success();
}

::mlir::LogicalResult AllocOp::verify() {
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

} // namespace mlir::tt::ttnn
