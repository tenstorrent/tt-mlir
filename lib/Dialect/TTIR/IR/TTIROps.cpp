// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.cpp.inc"
#include "ttmlir/Utils.h"

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

mlir::tt::ttir::ToLayoutOp::CompoundComponents
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
  bool isMemoryLayoutChange =
      inputLayout.getMemLayout() != outputLayout.getMemLayout();
  return {isLayoutChange, isGridChange, isFormatChange, isMemorySpaceChange,
          isMemoryLayoutChange};
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

  if (dim < 0) {
    dim += firstTensorRank;
  }

  // Check that the dimension `dim` is valid.
  if (dim < 0 || dim >= firstTensor.getRank()) {
    return emitOpError() << "Invalid dimension " << getDim()
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

::mlir::LogicalResult mlir::tt::ttir::MaxPool2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  std::vector<int64_t> inputShape = getInput().getType().getShape().vec();

  if (inputType.getRank() != 4) {
    return emitOpError()
           << "Input tensor rank must be 4. Recieved input with rank "
           << inputType.getRank() << ". Shape: (" << inputShape << ").";
  }

  if (getOriginalHeight().has_value() != getOriginalWidth().has_value()) {
    std::string with_value =
        getOriginalHeight().has_value() ? "original_height" : "original_width";
    return emitOpError()
           << "If providing the original height and width as attributes, both "
              "original_height and original_width must be set. However, only "
           << with_value << " was provided.";
  }

  if (getOriginalHeight().has_value() && getOriginalWidth().has_value()) {
    inputShape[1] = getOriginalHeight().value();
    inputShape[2] = getOriginalWidth().value();
  }

  if (getKernelHeight() > inputShape[1]) {
    return emitOpError() << "Kernel height " << getKernelHeight()
                         << " is greater than input height " << inputShape[1]
                         << ". This MaxPool2d configuration is invalid.";
  }

  if (getKernelWidth() > inputShape[2]) {
    return emitOpError() << "Kernel width " << getKernelWidth()
                         << " is greater than input width " << inputShape[2]
                         << ". This MaxPool2d configuration is invalid.";
  }

  return success();
}

::mlir::LogicalResult mlir::tt::ttir::SqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  int32_t dim = getDim();

  if (dim < 0) {
    dim += inputType.getRank();
  }

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

::mlir::LogicalResult mlir::tt::ttir::UnsqueezeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  int32_t dim = getDim();

  // Convert negative dim to its positive equivalent
  if (dim < 0) {
    dim += inputType.getRank() + 1;
  }

  // Check that the dim is within the bounds of the input tensor
  if (dim > inputType.getRank() || dim < 0) {
    return emitOpError(
        "Dimension attribute must be within the bounds of the input tensor");
  }

  // Check that the output tensor has one more dimension than the input tensor
  if (outputType.getRank() != inputType.getRank() + 1) {
    return emitOpError(
        "Output tensor must have one more dimension than the input tensor");
  }

  // and that the dimension added is of size 1
  if (outputType.getDimSize(dim) != 1) {
    return emitOpError("Dimension added must be of size 1");
  }

  // All dimensions of the input tensor must be the same as the output tensor
  // except for the dimension added
  for (int64_t i = 0, j = 0; i < outputType.getRank(); ++i) {
    if (i == dim) {
      continue;
    }

    if (inputType.getDimSize(j) != outputType.getDimSize(i)) {
      return emitOpError("All dimensions of the input tensor must be the same "
                         "as the output tensor except for the dimension added");
    }

    j++;
  }

  return success();
}

::mlir::OpFoldResult mlir::tt::ttir::ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

// ANCHOR: adding_an_op_matmul_ttir_verify
::mlir::LogicalResult mlir::tt::ttir::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimension for the
  // purpose of the matrix multiply. After the matrix multiply, the prepended
  // dimension is removed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimension for
  // the purpose of the matrix-vector product and removed after.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  }

  // Verify that the input A and input B has matching inner dimensions
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError(
        "Input A[-1](" + std::to_string(inputAShape[inputAShape.size() - 1]) +
        ") and B[-2](" + std::to_string(inputBShape[inputBShape.size() - 2]) +
        ") must have matching inner dimensions");
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct the
  // expected output shape
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims, inputBBatchDims;

    if (inputAShape.size() > 2) {
      inputABatchDims.insert(inputABatchDims.begin(), inputAShape.begin(),
                             inputAShape.end() - 2);
    }

    if (inputBShape.size() > 2) {
      inputBBatchDims.insert(inputBBatchDims.begin(), inputBShape.begin(),
                             inputBShape.end() - 2);
    }

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(inputABatchDims, inputBBatchDims,
                                            broadcastedShape)) {

      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape
    expectedOutputShape.insert(expectedOutputShape.begin(),
                               broadcastedShape.begin(),
                               broadcastedShape.end());
  }

  // Insert the input A and B inner dimensions in expected output shape
  // Consider the case where input A and B are vectors. In that case,
  // the dimension 1 is ommited from the output shape.
  if (inputAType.getRank() > 1) {
    expectedOutputShape.push_back(inputAShape[inputAShape.size() - 2]);
  }

  if (inputBType.getRank() > 1) {
    expectedOutputShape.push_back(inputBShape[inputBShape.size() - 1]);
  }

  // Check the case of a vector-vector product. At this moment we don't support
  // scalars in IR, hence check that the output is at least 1D tensor of size 1.
  if (expectedOutputShape.size() == 0) {
    if (outputType.getRank() < 1) {
      return emitOpError("Scalar output is not supported, output must be at "
                         "least a 1D tensor");
    }

    if (outputType.getRank() > 1 || outputType.getShape()[0] != 1) {
      return emitOpError("Scalar output must be a 1D tensor of size 1");
    }

    return llvm::success();
  }

  // Verify that the output shape is correct
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(" +
                       std::to_string(outputShape.size()) +
                       ") must match the expected output shape rank(" +
                       std::to_string(expectedOutputShape.size()) + ")");
  }

  // Verify each dim of the output shape
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (outputShape[i] != expectedOutputShape[i]) {
      return emitOpError(
          "Output shape dimension[" + std::to_string(i) + "](" +
          std::to_string(outputShape[i]) +
          ") doesn't match the expected output shape dimension[" +
          std::to_string(i) + "](" + std::to_string(expectedOutputShape[i]) +
          ")");
    }
  }

  return success();
}
// ANCHOR_END: adding_an_op_matmul_ttir_verify

::mlir::LogicalResult mlir::tt::ttir::Conv2dOp::verify() {
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
  }
  return success();
}

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

// InferType

// ::llvm::LogicalResult
// mlir::tt::ttir::ReshapeOp::inferReturnTypes(::mlir::MLIRContext *context,
//                                               ::std::optional<::mlir::Location>
//                                               location,
//                                               ::mlir::ValueRange operands,
//                                               ::mlir::DictionaryAttr
//                                               attributes,
//                                               ::mlir::OpaqueProperties
//                                               properties,
//                                               ::mlir::RegionRange regions,
//                                               ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes)
//                                               {

//    std::vector<int64_t> shape_vec;
//    mlir::tt::ttir::ReshapeOp::Adaptor adaptor(operands, attributes,
//    properties, regions); for (int64_t dim :
//    mlir::cast<RankedTensorType>(operands[0].getType()).getShape()) {
//     shape_vec.push_back(dim);
//    }

//    ArrayRef<int64_t> new_shape_i64(shape_vec);

//    auto input_type = mlir::cast<RankedTensorType>(operands[0].getType());
//    inferredReturnTypes.push_back(input_type.cloneWith(new_shape_i64,
//    input_type.getElementType())); return success();
// }

// ::llvm::LogicalResult
// mlir::tt::ttir::MeanOp::inferReturnTypes(::mlir::MLIRContext *context,
//                                               ::std::optional<::mlir::Location>
//                                               location,
//                                               ::mlir::ValueRange operands,
//                                               ::mlir::DictionaryAttr
//                                               attributes,
//                                               ::mlir::OpaqueProperties
//                                               properties,
//                                               ::mlir::RegionRange regions,
//                                               ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes)
//                                               {

//    mlir::tt::ttir::MeanOp::Adaptor adaptor(operands, attributes, properties,
//    regions);

//    std::vector<int64_t> shape_vec;
//    for (int64_t dim :
//    mlir::cast<RankedTensorType>(operands[0].getType()).getShape()) {
//     shape_vec.push_back(dim);
//    }
//    auto outty = mlir::cast<RankedTensorType>(operands[1].getType());
//    (void)outty;

//    bool keepdim = true; //attributes.contains("keep_dim");

//    if (adaptor.getDimArg().has_value()) {
//     ArrayRef reduce_dims_ = adaptor.getDimArg()->getValue();
//     std::vector<int32_t> reduce_dims;
//     for (Attribute dim_attr : reduce_dims_) {
//       int32_t dim = mlir::cast<IntegerAttr>(dim_attr).getInt();
//       if (dim < 0) {
//         dim += shape_vec.size();
//       }
//       reduce_dims.push_back(dim);
//     }

//     if (keepdim) {
//       for (int32_t dim : reduce_dims) {
//         shape_vec[dim] = 1;
//       }
//     } else {
//       std::sort(reduce_dims.begin(), reduce_dims.end());
//       for (int32_t dim : reduce_dims) {
//         shape_vec.erase(shape_vec.begin() + dim);
//       }
//       // In the event the reduce brings the tensor to a single value, make
//       sure shape is still [1] if (shape_vec.size() == 0) {
//         shape_vec = {1};
//       }
//     }

//    } else {
//     if (keepdim) {
//       for (uint64_t i = 0; i < shape_vec.size(); i++) {
//         shape_vec[i] = 1;
//       }
//     } else {
//       shape_vec = {1};
//     }
//    }

//     auto input_type = mlir::cast<RankedTensorType>(operands[0].getType());
//     inferredReturnTypes.push_back(input_type.cloneWith(shape_vec,
//     input_type.getElementType()));
//    return success();
// }

// ::llvm::LogicalResult
// mlir::tt::ttir::MaxOp::inferReturnTypes(::mlir::MLIRContext *context,
//                                               ::std::optional<::mlir::Location>
//                                               location,
//                                               ::mlir::ValueRange operands,
//                                               ::mlir::DictionaryAttr
//                                               attributes,
//                                               ::mlir::OpaqueProperties
//                                               properties,
//                                               ::mlir::RegionRange regions,
//                                               ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes)
//                                               {

//    std::vector<int64_t> shape_vec;
//    for (int64_t dim :
//    mlir::cast<RankedTensorType>(operands[0].getType()).getShape()) {
//     shape_vec.push_back(dim);
//    }
//    bool keepdim = attributes.getAs<BoolAttr>("keep_dim").getValue();

//    if (attributes.getAs<ArrayAttr>("dim_arg")) {
//     int32_t reduce_dim =
//     mlir::cast<IntegerAttr>(attributes.getAs<ArrayAttr>("dim_arg")[0]).getInt();
//     if (reduce_dim < 0) {
//       reduce_dim += shape_vec.size();
//     }

//     if (keepdim) {
//       shape_vec[reduce_dim] = 1;
//     } else {
//       shape_vec.erase(shape_vec.begin() + reduce_dim);
//     }

//    } else {
//     if (keepdim) {
//       for (uint64_t i = 0; i < shape_vec.size(); i++) {
//         shape_vec[i] = 1;
//       }
//     } else {
//       shape_vec = {1};
//     }
//    }

//     auto input_type = mlir::cast<RankedTensorType>(operands[0].getType());
//     inferredReturnTypes.push_back(input_type.cloneWith(shape_vec,
//     input_type.getElementType()));
//    return success();
// }

// ::llvm::LogicalResult
// mlir::tt::ttir::SumOp::inferReturnTypes(::mlir::MLIRContext *context,
//                                               ::std::optional<::mlir::Location>
//                                               location,
//                                               ::mlir::ValueRange operands,
//                                               ::mlir::DictionaryAttr
//                                               attributes,
//                                               ::mlir::OpaqueProperties
//                                               properties,
//                                               ::mlir::RegionRange regions,
//                                               ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes)
//                                               {

//    std::vector<int64_t> shape_vec;
//    for (int64_t dim :
//    mlir::cast<RankedTensorType>(operands[0].getType()).getShape()) {
//     shape_vec.push_back(dim);
//    }
//    bool keepdim = attributes.getAs<BoolAttr>("keep_dim").getValue();

//    if (attributes.getAs<ArrayAttr>("dim_arg")) {
//     int32_t reduce_dim =
//     mlir::cast<IntegerAttr>(attributes.getAs<ArrayAttr>("dim_arg")[0]).getInt();
//     if (reduce_dim < 0) {
//       reduce_dim += shape_vec.size();
//     }

//     if (keepdim) {
//       shape_vec[reduce_dim] = 1;
//     } else {
//       shape_vec.erase(shape_vec.begin() + reduce_dim);
//     }

//    } else {
//     if (keepdim) {
//       for (uint64_t i = 0; i < shape_vec.size(); i++) {
//         shape_vec[i] = 1;
//       }
//     } else {
//       shape_vec = {1};
//     }
//    }

//     auto input_type = mlir::cast<RankedTensorType>(operands[0].getType());
//     inferredReturnTypes.push_back(input_type.cloneWith(shape_vec,
//     input_type.getElementType()));
//    return success();
// }
