// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/TTIRDialect.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/TTIROps.cpp.inc"

::mlir::LogicalResult mlir::tt::ttir::LayoutOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto inputLayout =
      inputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  auto outputLayout =
      outputTy.getEncoding().template dyn_cast_or_null<mlir::tt::LayoutAttr>();
  if (getIsCast() and inputLayout and outputLayout) {
    auto inputMemref = inputLayout.getMemref();
    auto outputMemref = outputLayout.getMemref();
    // If IsCast is true, the input and output memspace must match
    if (inputMemref.getMemorySpace() != outputMemref.getMemorySpace()) {
      return failure();
    }
  } else if (bool(inputLayout) != bool(outputLayout)) {
    // If transitioning to/from a layout to a non-layout, the memspace must be
    // system
    auto memref =
        inputLayout ? inputLayout.getMemref() : outputLayout.getMemref();
    if (memref.getMemorySpace()
            .template cast<mlir::tt::MemorySpaceAttr>()
            .getValue() != mlir::tt::MemorySpace::System) {
      return failure();
    }
  }
  return success();
}
