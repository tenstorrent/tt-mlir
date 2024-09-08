// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include <optional>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.cpp.inc"

namespace mlir::tt::ttnn {

constexpr int TTNN_TILE_HEIGHT = 32;
constexpr int TTNN_TILE_WIDTH = 32;

static bool isValidDeviceLayout(::mlir::tt::TensorMemoryLayout layout) {
  return layout == ::mlir::tt::TensorMemoryLayout::Interleaved ||
         ::mlir::tt::isShardedMemoryLayout(layout);
}

::mlir::LogicalResult mlir::tt::ttnn::ToMemoryConfigOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getResult().getType();
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
  ::mlir::tt::MemorySpace outputMemorySpace = outputLayout.getMemorySpace();
  ::mlir::tt::TensorMemoryLayout outputMemoryLayout =
      outputLayout.getMemLayout();
  if (::mlir::tt::isSystemMemorySpace(outputMemorySpace) &&
      outputMemoryLayout != ::mlir::tt::TensorMemoryLayout::None) {
    return emitOpError("System memory space only supports undef memory layout");
  }

  if (::mlir::tt::isDeviceMemorySpace(outputMemorySpace) &&
      !isValidDeviceLayout(outputMemoryLayout)) {
    return emitOpError("Device memory space only supports interleaved or "
                       "sharded memory layouts");
  }

  if (outputMemorySpace == ::mlir::tt::MemorySpace::DeviceDRAM &&
      outputMemoryLayout != ::mlir::tt::TensorMemoryLayout::Interleaved) {
    return emitOpError(
        "Device DRAM memory space only supports interleaved memory layout");
  }

  if (outputLayout.hasShardedTensorMemoryLayout()) {
    if (outputMemoryLayout != ::mlir::tt::TensorMemoryLayout::BlockSharded) {
      return emitOpError("Currently only block sharding is supported for "
                         "sharded memory layouts");
    }
    ::llvm::SmallVector<int64_t> shardShape = outputLayout.getShardShape();
    // Currently TTNN backend only supports 2D shard shape
    if (shardShape.size() != 2) {
      return emitOpError("Shard shape must be 2D");
    }
    // TTNN tiles are (32, 32), shard shape must evenly divide the tile shape
    if (shardShape[0] % TTNN_TILE_HEIGHT != 0 or
        shardShape[1] % TTNN_TILE_WIDTH != 0) {
      return emitOpError("Shard shape must divide tile shape (32, 32) evenly");
    }
  }
  return success();
}

::mlir::LogicalResult mlir::tt::ttnn::EmbeddingOp::verify() {
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

::mlir::LogicalResult mlir::tt::ttnn::SoftmaxOp::verify() {
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

::mlir::LogicalResult mlir::tt::ttnn::TransposeOp::verify() {
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

::mlir::LogicalResult mlir::tt::ttnn::ConcatOp::verify() {
  mlir::OperandRange inputs = getInputs();
  int32_t dim = getAxis();
  mlir::RankedTensorType firstTensor =
      mlir::cast<mlir::RankedTensorType>(inputs.front().getType());
  int64_t firstTensorRank = firstTensor.getRank();

  if (dim < 0) {
    dim += firstTensorRank;
  }

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= firstTensor.getRank()) {
    return emitOpError() << "Invalid dimension " << getAxis()
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

  return mlir::success();
}

::mlir::LogicalResult mlir::tt::ttnn::ReshapeOp::verify() {
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

// ANCHOR: adding_an_op_matmul_ttnn_verify
::mlir::LogicalResult mlir::tt::ttnn::MatmulOp::verify() {
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
// ANCHOR_END: adding_an_op_matmul_ttnn_verify

::mlir::LogicalResult mlir::tt::ttnn::Conv2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  std::optional<::mlir::RankedTensorType> biasType =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  if (inputType.getRank() < 3) {
    return emitOpError("Input must be at least a 3D tensor");
  }
  if (weightType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }
  if (biasType.has_value()) {
    if (biasType->getRank() != 4) {
      return emitOpError("Bias must be a 4D tensor");
    }
    auto biasShape = biasType->getShape();
    if (biasShape[0] != 1 || biasShape[1] != 1 || biasShape[2] != 1) {
      return emitOpError("Bias must only have data on the final dimenstion");
    }
  }
  return success();
}

::mlir::LogicalResult AllocOp::verify() {
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

} // namespace mlir::tt::ttnn
