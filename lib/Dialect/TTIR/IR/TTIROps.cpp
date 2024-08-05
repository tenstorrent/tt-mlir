// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.cpp.inc"
#include <mlir/IR/BuiltinTypes.h>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"

::mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getOutput().getType();
  auto inputLayout =
      mlir::dyn_cast_or_null<mlir::tt::LayoutAttr>(inputTy.getEncoding());
  auto outputLayout =
      mlir::dyn_cast_or_null<mlir::tt::LayoutAttr>(outputTy.getEncoding());
  if (not inputLayout) {
    return emitOpError("Input tensor type missing layout attribute");
  }
  if (not outputLayout) {
    return emitOpError("Output tensor type missing layout attribute");
  }
  return success();
}

::mlir::LogicalResult mlir::tt::ttir::SoftmaxOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

  // Shapes of input and output of a softmax operation must be the same
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("Input and output shapes must be the same");
  }

  int32_t dim = getDimension();

  // Check that the dim is within the bounds of the input tensor
  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError(
        "Dimension attribute must be within the bounds of the input tensor");
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
  auto layout = mlir::dyn_cast_or_null<mlir::tt::LayoutAttr>(
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

::mlir::LogicalResult mlir::tt::ttir::ConstantOp::verify() {
    // // Check if the result is a tensor type
    // auto tensorType = mlir::dyn_cast_or_null<RankedTensorType>(getResult().getType());
    // if (!tensorType) {
    //     return emitError("Result must be of ranked tensor type");
    // }

    // // Ensure the tensor is not empty.
    // if (mlir::isa<mlir::NoneType>(tensorType.getElementType())) {
    //     return emitError("Result tensor cannot have NoneType element type");
    // }

    // // Get the value attribute
    // auto valueAttr = getValue();
    // if (!valueAttr) {
    //     return emitError("value attribute must be specified");
    // }

    // // Check if valueAttr is an integer or float
    // if (valueAttr.isa<IntegerAttr>()) {
    //     auto intAttr = valueAttr.cast<IntegerAttr>();
    //     auto elementType = tensorType.getElementType().dyn_cast<IntegerType>();
    //     if (!elementType) {
    //         return emitError("result tensor must have integer element type for integer constants");
    //     }
    //     if (intAttr.getType() != elementType) {
    //         return emitError("value type does not match tensor element type for integer constants");
    //     }
    // } else if (valueAttr.isa<FloatAttr>()) {
    //     auto floatAttr = valueAttr.cast<FloatAttr>();
    //     auto elementType = tensorType.getElementType().dyn_cast<FloatType>();
    //     if (!elementType) {
    //         return emitError("result tensor must have float element type for float constants");
    //     }
    //     if (floatAttr.getType() != elementType) {
    //         return emitError("value type does not match tensor element type for float constants");
    //     }
    // } else {
    //     return emitError("value attribute must be either an integer or float");
    // }

    return success();
}