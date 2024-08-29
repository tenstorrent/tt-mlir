// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

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
  if (inputTy.getShape() != outputTy.getShape()) {
    return emitOpError("Input and output shapes must be the same");
  }
  return success();
}

std::tuple<bool, bool, bool, bool>
mlir::tt::ttir::ToLayoutOp::compoundComponents() {
  auto inputLayout =
      mlir::cast<tt::LayoutAttr>(getInput().getType().getEncoding());
  auto outputLayout =
      mlir::cast<tt::LayoutAttr>(getOutput().getType().getEncoding());
  bool isLayoutChange = inputLayout.getLinear() != outputLayout.getLinear();
  bool isGridChange = inputLayout.getGrid() != outputLayout.getGrid();
  bool isShardChange =
      inputLayout.getShardShape() != outputLayout.getShardShape();
  assert(isGridChange == isShardChange);
  bool isFormatChange =
      inputLayout.getElementType() != outputLayout.getElementType();
  bool isMemorySpaceChange =
      inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
  return std::make_tuple(isLayoutChange, isGridChange, isFormatChange,
                         isMemorySpaceChange);
}

::mlir::LogicalResult mlir::tt::ttir::GenericOp::verify() {
  if (getNumOperands() != getRegion().getNumArguments()) {
    return emitOpError(
        "The number of op operands and region/block operands must match");
  }
  return success();
}

template <typename OpTy>
static void buildGenericEltwiseBinaryRegion(::mlir::Location loc,
                                            ::mlir::OpBuilder &opBuilder,
                                            ::mlir::Block *block) {
  auto lhs = block->getArgument(0);
  auto rhs = block->getArgument(1);
  auto result = opBuilder.create<OpTy>(loc, lhs, rhs);
  opBuilder.create<mlir::tt::ttir::YieldOp>(loc, mlir::ValueRange({result}));
}

void mlir::tt::ttir::AddOp::buildGenericRegion(::mlir::OpBuilder &opBuilder,
                                               ::mlir::Block *block) {
  return buildGenericEltwiseBinaryRegion<arith::AddFOp>(getLoc(), opBuilder,
                                                        block);
}

void mlir::tt::ttir::MultiplyOp::buildGenericRegion(
    ::mlir::OpBuilder &opBuilder, ::mlir::Block *block) {
  return buildGenericEltwiseBinaryRegion<arith::MulFOp>(getLoc(), opBuilder,
                                                        block);
}

::mlir::LogicalResult mlir::tt::ttir::EmbeddingOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

  // inputType can have any rank

  // weightType must have rank of 2: (dictionary_size, embedding_size)
  //
  if (weightType.getRank() != 2) {
    return emitOpError("Weight must be a 2D tensor");
  }

  // outputType must have rank of inputType + and additional dimension of
  // embedding_size
  //
  if (outputType.getRank() - inputType.getRank() != 1) {
    return emitOpError("Output must have one dimension more than input");
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

::mlir::LogicalResult mlir::tt::ttir::TransposeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();
  int32_t dim0 = getDim0();
  int32_t dim1 = getDim1();
  if (inputType.getRank() < 2) {
    return emitOpError("Input must be at least a 2D tensor");
  }
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError("Input must have the same rank as output");
  }
  if (dim0 >= inputType.getRank() || dim0 < -inputType.getRank()) {
    return emitOpError(
        "Dimension 0 attribute must be within the bounds of the input tensor");
  }
  if (dim1 >= inputType.getRank() || dim1 < -inputType.getRank()) {
    return emitOpError(
        "Dimension 1 attribute must be within the bounds of the input tensor");
  }
  if (dim0 < 0) {
    dim0 += inputType.getRank();
  }
  if (dim1 < 0) {
    dim1 += inputType.getRank();
  }
  if (outputShape[dim0] != inputShape[dim1] ||
      outputShape[dim1] != inputShape[dim0]) {
    return emitOpError("Input-output transpose dimension mismatch.");
  }
  return success();
}

::mlir::LogicalResult mlir::tt::ttir::ConcatOp::verify() {
  mlir::OperandRange inputs = getInputs();
  int32_t dim = getDim();
  mlir::RankedTensorType firstTensor =
      mlir::cast<mlir::RankedTensorType>(inputs.front().getType());
  int64_t firstTensorRank = firstTensor.getRank();

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= firstTensor.getRank()) {
    return emitOpError() << "Invalid dimension " << dim
                         << " for concatenation.";
  }

  // Get the rank of the first input tensor
  // and check that all input tensors have the same rank
  // and that all dimensions except `dim` are the same.
  for (auto input : inputs.drop_front()) {
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    // Check if all inputs have the same rank.
    if (inputType.getRank() != firstTensorRank) {
      return emitOpError("All input tensors must have the same rank.");
    }

    // Check that dimensions (except `dim`) are the same.
    for (int64_t i = 0; i < firstTensorRank; ++i) {
      if (i != dim && inputType.getDimSize(i) != firstTensor.getDimSize(i)) {
        return emitOpError() << "All input tensors must have the same "
                                "dimensions, except for dimension "
                             << dim << ".";
      }
    }
  }

  return success();
}

::mlir::LogicalResult mlir::tt::ttir::ReshapeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  auto shape = getShape();
  int64_t shape_size = static_cast<int64_t>(shape.size());

  // Check that the shape size matches the rank of the output tensor
  if (shape_size != static_cast<int64_t>(outputType.getRank())) {
    return emitOpError("Shape attribute size must match output tensor rank");
  }

  // Check that the shape attribute is non-empty
  if (shape_size == 0) {
    return emitOpError("Shape attribute must be non-empty");
  }

  // Check that the shape attribute has at most 5 elements
  if (shape_size > 5) {
    return emitOpError("Shape attribute must have at most 5 elements");
  }

  // Cardinality of the input and output tensors must be the same
  if (inputType.getNumElements() != outputType.getNumElements()) {
    return emitOpError(
        "Input and output tensors must have the same number of elements");
  }

  bool has_negative = false;
  int64_t known_dim_product = 1;
  auto outputShape = outputType.getShape();

  // Check that all dimensions are positive except for at most one -1
  // Check that the non-negative dimensions match the output tensor shape
  // Calculate the product of the known dimensions
  for (int64_t i = 0; i < shape_size; i++) {
    int64_t dim_value = mlir::cast<IntegerAttr>(shape[i]).getInt();

    if (dim_value == -1) {
      if (has_negative) {
        return emitOpError("Shape attribute must have at most one -1 element");
      }
      has_negative = true;
    } else {
      if (dim_value <= 0) {
        return emitOpError(
            "All dimensions must be positive except the one with -1");
      }

      // Ensure that the non-negative dimensions match the output tensor shape
      if (dim_value != outputShape[i]) {
        return emitOpError("Shape attribute must match the output tensor shape "
                           "for dimensions that are not -1");
      }

      known_dim_product *= dim_value;
    }
  }

  // If there's a -1, ensure that it can be inferred correctly
  if (has_negative && inputType.getNumElements() % known_dim_product != 0) {
    return emitOpError("Invalid shape: the dimensions do not multiply to the "
                       "total number of elements in the tensor");
  }

  return success();
}

::mlir::LogicalResult mlir::tt::ttir::SqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  int32_t dim = getDim();

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension " << dim << " for squeezing.";
  }

  // Check that the dimension `dim` is 1 in the input tensor.
  if (inputType.getDimSize(dim) != 1) {
    return emitOpError() << "Dimension " << dim
                         << " in the input tensor must be 1.";
  }

  if (outputType.getRank() == 0) {
    return emitOpError() << "Output tensor must have at least one dimension.";
  }

  // Check that the rank of the output tensor is one less than the input tensor.
  if (outputType.getRank() != inputType.getRank() - 1) {
    return emitOpError()
           << "Output tensor rank must be one less than the input tensor rank.";
  }

  // Check that the dimensions of the output tensor are the same as the input
  // tensor except for dimension `dim`.
  for (int64_t i = 0, j = 0; i < inputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }
    if (inputType.getDimSize(i) != outputType.getDimSize(j)) {
      return emitOpError() << "Dimensions of the output tensor must be the "
                              "same as the input tensor except for dimension "
                           << dim << ".";
    }
    ++j;
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
