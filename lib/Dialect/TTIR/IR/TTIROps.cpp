// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.cpp.inc"
#include <unordered_set>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"

::mlir::LogicalResult mlir::tt::ttir::LayoutOp::verify() {
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

// ANCHOR: adding_an_op_matmul_ttir_verify
::mlir::LogicalResult mlir::tt::ttir::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  auto inputAShape = inputAType.getShape();
  auto inputBShape = inputBType.getShape();
  auto outputShape = outputType.getShape();
  if (inputAShape.size() < 2) {
    return emitOpError("Input A must be at least a 2D tensor");
  }
  if (inputBShape.size() < 2) {
    return emitOpError("Input B must be at least a 2D tensor");
  }
  if (inputAShape.size() != inputBShape.size()) {
    return emitOpError("Input A and B must have the same rank");
  }
  if (inputAShape.size() != outputShape.size()) {
    return emitOpError("Input A and B must have the same rank as the output");
  }
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A and B must have matching inner dimensions");
  }
  if (outputShape[outputShape.size() - 2] !=
      inputAShape[inputAShape.size() - 2]) {
    return emitOpError("Output must have the same number of rows as input A");
  }
  if (outputShape[outputShape.size() - 1] !=
      inputBShape[inputBShape.size() - 1]) {
    return emitOpError(
        "Output must have the same number of columns as input B");
  }
  return success();
}
// ANCHOR_END: adding_an_op_matmul_ttir_verify

::mlir::LogicalResult mlir::tt::ttir::AllocOp::verify() {
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

::mlir::LogicalResult mlir::tt::ttir::SumOp::verify() {
  auto keepDim = getKeepDim();
  if (not keepDim) {
    return emitOpError("Sum op must have keep dim attribute set to true");
  }

  auto inputShape = getInput().getType().getShape();
  auto dimArgOpt = getDimArg();
  if (dimArgOpt.has_value()) {
    const auto &dimArgArray = dimArgOpt.value();
    if (dimArgArray.size() > inputShape.size()) {
      return emitOpError("Dim arg attribute has more indexes then there are "
                         "dimensions of the input tensor");
    }

    const int maxPositiveIndex = inputShape.size() - 1;
    const int minNegativeIndex = -inputShape.size();
    std::unordered_set<int64_t> seenIndexes;
    for (auto reductionAxisAttr : dimArgArray) {
      auto reductionAxis = reductionAxisAttr.cast<IntegerAttr>().getInt();

      if (seenIndexes.count(reductionAxis)) {
        return emitOpError("Duplicate reduction axis");
      }
      seenIndexes.insert(reductionAxis);

      if (reductionAxis < minNegativeIndex ||
          reductionAxis > maxPositiveIndex) {
        return emitOpError("Reduction axis out of range");
      }
    }
  }

  return success();
}
