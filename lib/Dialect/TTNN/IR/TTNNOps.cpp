// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Traits.h"

#include <numeric>
#include <optional>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.cpp.inc"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::ClampOp::verify() {
  ::mlir::Operation::operand_range inputs = getInputs();
  ::mlir::Operation::result_range outputs = getResults();

  if (inputs.size() != 1) {
    return emitOpError("expects one tensor as input.");
  }

  if (outputs.size() != 1) {
    return emitOpError("generates one tensor as output.");
  }

  const RankedTensorType inputTensorType =
      mlir::cast<RankedTensorType>(inputs.front().getType());

  const RankedTensorType outputTensorType =
      mlir::cast<RankedTensorType>(outputs.front().getType());

  if (inputTensorType.getShape() != outputTensorType.getShape()) {
    return emitOpError("input and output must have same shape.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

// Conv2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::Conv2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getResult().getType();
  std::optional<mlir::RankedTensorType> bias =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  if (inputType.getRank() != 4) {
    return emitOpError("Input must be a 4D tensor");
  }

  if (outputType.getRank() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  if (weightType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }

  if (bias.has_value()) {
    if (bias->getRank() != 4) {
      return emitOpError("Bias must be a 4D tensor");
    }
    auto biasShape = bias->getShape();
    if (biasShape[0] != 1 || biasShape[1] != 1 || biasShape[2] != 1) {
      return emitOpError("Bias must only have data on the final dimenstion");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

// ConvTranspose2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::ConvTranspose2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getResult().getType();
  std::optional<mlir::RankedTensorType> bias =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  if (inputType.getRank() != 4) {
    return emitOpError("Input must be a 4D tensor");
  }

  if (outputType.getRank() != 4) {
    return emitOpError("Output must be a 4D tensor");
  }

  if (weightType.getRank() != 4) {
    return emitOpError("Weight must be a 4D tensor");
  }

  if (bias.has_value()) {
    if (bias->getRank() != 4) {
      return emitOpError("Bias must be a 4D tensor");
    }
  }

  std::function<mlir::LogicalResult(llvm::ArrayRef<int32_t> &, const char *,
                                    int32_t)>
      checkBiggerThan = [&](llvm::ArrayRef<int32_t> &values, const char *name,
                            int32_t minValue) -> mlir::LogicalResult {
    for (int32_t value : values) {
      if (value < minValue) {
        return emitOpError() << "Attribute '" << name
                             << "' contains a value less than: " << minValue;
      }
    }
    return mlir::success();
  };

  uint32_t inChannels = getInChannels();
  if (inChannels != inputType.getDimSize(inputType.getRank() - 1)) {
    return emitOpError("Input channels attribute must match "
                       "the last dimension of the input tensor");
  }

  uint32_t outChannels = getOutChannels();
  if (outChannels != outputType.getDimSize(outputType.getRank() - 1)) {
    return emitOpError("Output channels attribute match "
                       "the last dimension of the output tensor");
  }

  uint32_t batchSize = getBatchSize();
  if (batchSize != inputType.getDimSize(0)) {
    return emitOpError("Batch size attribute must match the first "
                       "dimension of the input tensor");
  }

  uint32_t inputHeight = getInputHeight();
  if (inputHeight != inputType.getDimSize(inputType.getRank() - 3)) {
    return emitOpError("Input height attribute must match the second "
                       "dimension of the input tensor");
  }

  uint32_t inputWidth = getInputWidth();
  if (inputWidth != inputType.getDimSize(inputType.getRank() - 2)) {
    return emitOpError("Input width attribute must match the third "
                       "dimension of the input tensor");
  }

  llvm::ArrayRef<int32_t> stride = getStride();
  if (failed(checkBiggerThan(stride, "stride", 1))) {
    return mlir::failure();
  }

  llvm::ArrayRef<int32_t> padding = getPadding();
  if (failed(checkBiggerThan(padding, "padding", 0))) {
    return mlir::failure();
  }

  llvm::ArrayRef<int32_t> outputPadding = getOutputPadding();
  if (failed(checkBiggerThan(outputPadding, "output padding", 0))) {
    return mlir::failure();
  }

  llvm::ArrayRef<int32_t> dilation = getDilation();
  if (failed(checkBiggerThan(dilation, "dilation", 1))) {
    return mlir::failure();
  }

  llvm::ArrayRef<std::int64_t> kernelShape = weightType.getShape();

  int32_t inputChannels = inputType.getDimSize(inputType.getRank() - 1);
  int32_t outputChannels = outputType.getDimSize(outputType.getRank() - 1);
  uint32_t groups = getGroups();

  if (inputChannels % groups != 0) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << inputChannels << " input channels and "
                         << groups << " groups.";
  }

  if (outputChannels % groups != 0) {
    return emitOpError() << "Number of output channels from output tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << outputChannels << " output channels and "
                         << groups << " groups.";
  }

  if (inputChannels != kernelShape[0]) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "match the first dimension of the weight tensor. "
                         << "Got " << inputChannels << " input channels and "
                         << kernelShape[0] << " in the weight tensor.";
  }

  if (outputChannels / groups != kernelShape[1]) {
    return emitOpError() << "Number of output channels per group must match "
                            "the second dimension of the weight tensor. "
                         << "Got " << (outputChannels / groups)
                         << " output channels per group and " << kernelShape[1]
                         << " in the weight tensor.";
  }

  if (bias) {
    if (bias->getDimSize(bias->getRank() - 1) != outputChannels) {
      return emitOpError() << "Mismatch in bias tensor dimensions. "
                           << "Bias tensor has "
                           << bias->getDimSize(bias->getRank() - 1)
                           << " channels, "
                           << "but the output tensor has " << outputChannels
                           << " channels.";
    }
  }

  int32_t kernelHeight = kernelShape[2];
  int32_t kernelWidth = kernelShape[3];

  int32_t Hin = inputType.getDimSize(inputType.getRank() - 3);
  int32_t Win = inputType.getDimSize(inputType.getRank() - 2);

  int32_t expectedHOut = (Hin - 1) * stride[0] - 2 * padding[0] +
                         dilation[0] * (kernelHeight - 1) + outputPadding[0] +
                         1;
  int32_t expectedWOut = (Win - 1) * stride[1] - 2 * padding[1] +
                         dilation[1] * (kernelWidth - 1) + outputPadding[1] + 1;
  if (expectedHOut < 0 || expectedWOut < 0) {
    return emitOpError() << "Given input size per channel: (" << Hin << " x "
                         << Win << "). "
                         << "Calculated output size per channel: ("
                         << expectedHOut << " x " << expectedWOut << "). "
                         << "Output size is too small";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ToDTypeOp
//===----------------------------------------------------------------------===//

// ToDTypeOp folder
::mlir::OpFoldResult mlir::tt::ttnn::ToDTypeOp::fold(FoldAdaptor adaptor) {

  // If the input and output are same, fold to the input.
  if (getType() == getInput().getType()) {
    return getInput();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

// MaxPool2dOp verification
::mlir::LogicalResult mlir::tt::ttnn::MaxPool2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();

  if (getKernelHeight() > getInputHeight()) {
    return emitOpError() << "Kernel height " << getKernelHeight()
                         << " is greater than input height " << getInputHeight()
                         << ". This MaxPool2d configuration is invalid.";
  }

  if (getKernelWidth() > getInputWidth()) {
    return emitOpError() << "Kernel width " << getKernelWidth()
                         << " is greater than input width " << getInputWidth()
                         << ". This MaxPool2d configuration is invalid.";
  }

  if (inputType.getRank() != 4) {
    return emitOpError()
           << "Input tensor rank must be 4. Received input with rank "
           << inputType.getRank() << ". Shape: (" << inputShape << ").";
  }

  if (inputShape[0] != 1 || inputShape[1] != 1) {
    return emitOpError() << "Maxpool input must be in the form (1, 1, N*H*W, "
                            "C). Received shape ("
                         << inputShape << ").";
  }

  if (inputShape[2] != getBatchSize() * getInputHeight() * getInputWidth()) {
    return emitOpError() << "Maxpool shape (" << inputShape
                         << ") at dim -2 must be equal to N*H*W. However the "
                            "attributes given are N="
                         << getBatchSize() << ", H=" << getInputHeight()
                         << ", W=" << getInputWidth() << ". " << getBatchSize()
                         << "*" << getInputHeight() << "*" << getInputWidth()
                         << " != " << inputShape[2] << ".";
  }

  if (inputShape[3] != getChannels()) {
    return emitOpError() << "Maxpool shape (" << inputShape
                         << ") at dim -3 must be equal to C. However the "
                            "attribute given is C="
                         << getChannels() << ". " << inputShape[3]
                         << " != " << getChannels();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttnn::ArangeOp::verify() {

  if (getStep() == 0) {
    return emitOpError("Step cannot be zero.");
  }

  int64_t numValues = (getEnd() - getStart()) / getStep();

  if (numValues <= 0) {
    return emitOpError("Invalid range: start=")
           << getStart() << ", end=" << getEnd() << ", step=" << getStep();
  }

  std::vector<int64_t> expectedShape = {1, 1, 1, numValues};
  if (getType().getShape().vec() != expectedShape) {
    return emitOpError() << "Output tensor shape must be " << expectedShape
                         << ", but got " << getType().getShape();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//

// EmptyOp verification
::mlir::LogicalResult mlir::tt::ttnn::EmptyOp::verify() {
  // Check that the attributes of the op match the attributes of the output
  // tensor type.
  //
  assert(::llvm::isa<RankedTensorType>(getResult().getType()));
  RankedTensorType output = mlir::cast<RankedTensorType>(getResult().getType());

  assert(::llvm::isa<TTNNLayoutAttr>(output.getEncoding()));
  TTNNLayoutAttr layoutAttr = mlir::cast<TTNNLayoutAttr>(output.getEncoding());

  // Shape
  //
  assert(output.getShape() == getShape().getShape());

  // DataType and Layout
  //
  assert(getLayout() == layoutAttr.getLayout());
  assert(getDtype() == layoutAttr.getDataType());

  // MemoryConfig
  // Compare internal attrs with output tensor attrs.
  //
  assert(getMemoryConfig().getBufferType().getValue() ==
         layoutAttr.getBufferType());
  assert(getMemoryConfig().getTensorMemoryLayout() ==
         layoutAttr.getMemLayout());

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// ConcatOp verification
::mlir::LogicalResult mlir::tt::ttnn::ConcatOp::verify() {
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

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

// RepeatOp verification
::mlir::LogicalResult mlir::tt::ttnn::RepeatOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  llvm::ArrayRef<int64_t> repeatDims = getRepeatDims().getShape();

  // Verify that the input tensor and repeat_dims argument have same rank.
  if (inputType.getRank() != static_cast<int64_t>(repeatDims.size())) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the number of repeat dimensions "
                         << repeatDims.size() << ".";
  }

  // Verify that the input and output tensor have same rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the output tensor rank "
                         << outputType.getRank() << ".";
  }

  // Verify expected output shape.
  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  for (size_t i = 0; i < repeatDims.size(); i++) {
    // Verify that the repeat dimension is greater than 0.
    if (repeatDims[i] <= 0) {
      return emitOpError() << "Repeat dimension at index " << i
                           << " must be greater than 0.";
    }

    int64_t dimValue = repeatDims[i];
    if (inputShape[i] * dimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",")
                           << ") at index " << i
                           << " does not repeat to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using repeat value " << dimValue << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// ReshapeOp verification
::mlir::LogicalResult mlir::tt::ttnn::ReshapeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  auto shape = getShape();
  int64_t shapeSize = static_cast<int64_t>(shape.size());

  // Check that the shape size matches the rank of the output tensor
  if (shapeSize != static_cast<int64_t>(outputType.getRank())) {
    return emitOpError("Shape attribute size must match output tensor rank");
  }
  // Check that the shape attribute is non-empty
  if (shapeSize == 0) {
    return emitOpError("Shape attribute must be non-empty");
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
  for (int64_t i = 0; i < shapeSize; i++) {
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

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// PadOp verification
::mlir::LogicalResult mlir::tt::ttnn::PadOp::verify() {

  ::mlir::RankedTensorType inputType = getInput().getType();

  // Check that size of padding is correct
  if (static_cast<int64_t>(getPadding().size()) != 2 * inputType.getRank()) {
    return emitOpError("Padding must have the same number of elements as twice "
                       "the rank of the input tensor");
  }

  std::vector<int64_t> inferredShapeVec = inputType.getShape().vec();
  llvm::ArrayRef<int32_t> padding = getPadding();
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    inferredShapeVec[i] += padding[2 * i];
    inferredShapeVec[i] += padding[2 * i + 1];
  }
  llvm::ArrayRef<int64_t> inferredShape = inferredShapeVec;

  // Check that the output tensor shape is correct
  ::mlir::RankedTensorType resultType = getResult().getType();
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  if (resultShape != inferredShape) {
    return emitOpError("Output tensor shape (" +
                       ttmlir::utils::join(resultShape, ",") +
                       ") must match the inferred shape: (" +
                       ttmlir::utils::join(inferredShape, ",") + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// SliceOp verification
::mlir::LogicalResult mlir::tt::ttnn::SliceOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::ArrayAttr begins = getBeginsAttr();
  ::mlir::ArrayAttr ends = getEndsAttr();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Verify that the input is at least 1D tensor
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Verify that the input rank matches number of elements in begins, ends, and
  // step
  size_t input_rank = static_cast<size_t>(inputType.getRank());
  if (input_rank != begins.size() || input_rank != ends.size() ||
      input_rank != stepAttr.size()) {
    return emitOpError("Begins, ends, and step attributes must have the same "
                       "number of elements as the input tensor rank");
  }

  // Validate that the output tensor has the same element type as the input
  // tensor
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError(
        "Output tensor must have the same element type as the input tensor");
  }

  // Verify the output tensor rank
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError(
        "Output tensor must have the same rank as the input tensor");
  }

  // Verify begin, end, step and the output tensor dimensions
  for (size_t i = 0; i < input_rank; ++i) {
    int64_t dimSize = inputShape[i];

    int32_t begin = ::mlir::cast<::mlir::IntegerAttr>(begins[i]).getInt();
    int32_t end = ::mlir::cast<::mlir::IntegerAttr>(ends[i]).getInt();
    int32_t step = ::mlir::cast<::mlir::IntegerAttr>(stepAttr[i]).getInt();

    // Adjust negative begin and end
    int32_t adjustedBegin = (begin < 0) ? (begin + dimSize) : begin;
    int32_t adjustedEnd = (end < 0) ? (end + dimSize) : end;

    std::ostringstream inputShapeStream;
    inputShapeStream << "(";
    for (size_t i = 0; i < inputShape.size(); ++i) {
      inputShapeStream << inputShape[i];
      if (i != inputShape.size() - 1) {
        inputShapeStream << ", ";
      }
    }
    inputShapeStream << ")";
    std::string inputShapeStr = inputShapeStream.str();

    if (adjustedBegin < 0 || adjustedBegin >= dimSize) {
      return emitOpError() << "Invalid begin index for dimension "
                           << std::to_string(i) << ". Expected value in range ["
                           << std::to_string(-dimSize) << ", " << dimSize
                           << "), got " << begin
                           << ". Input shape: " << inputShapeStr;
    }
    if (adjustedEnd < 0 || adjustedEnd > dimSize) {
      return emitOpError() << "Invalid end index for dimension "
                           << std::to_string(i) << ". Expected value in range ["
                           << std::to_string(-dimSize) << ", " << dimSize
                           << "], got " << end
                           << ". Input shape: " << inputShapeStr;
    }

    auto formatValueMessage = [](int value, int adjustedValue) {
      return value < 0 ? std::to_string(adjustedValue) + " (" +
                             std::to_string(value) + ")"
                       : std::to_string(value);
    };
    std::string beginValueMessage = formatValueMessage(begin, adjustedBegin);
    std::string endValueMessage = formatValueMessage(end, adjustedEnd);

    if (step == 0) {
      return emitOpError("Step value for dimension " + std::to_string(i) +
                         " cannot be zero");
    }

    if (step > 0 && adjustedBegin > adjustedEnd) {
      return emitOpError() << "For positive step, begin index must be less "
                              "than or equal to end index for dimension "
                           << i << ". Got begin: " << beginValueMessage
                           << ", end: " << endValueMessage << ", step: " << step
                           << ", input shape: " << inputShapeStr;
    }

    if (step < 0 && adjustedBegin < adjustedEnd) {
      return emitOpError() << "For negative step, begin index must be greater "
                              "than or equal to end index for dimension "
                           << i << ". Got begin: " << beginValueMessage
                           << ", end: " << endValueMessage << ", step: " << step
                           << ", input shape: " << inputShapeStr;
    }

    // Calculate the expected size of the output dimension
    int32_t expectedDimSize =
        (std::abs(adjustedEnd - adjustedBegin) + std::abs(step) - 1) /
        std::abs(step);
    if (outputType.getDimSize(i) != expectedDimSize) {
      return emitOpError() << "Mismatch in dimension " << std::to_string(i)
                           << " of the output tensor: expected size "
                           << expectedDimSize << ", but got "
                           << outputType.getDimSize(i);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// TransposeOp verification
::mlir::LogicalResult mlir::tt::ttnn::TransposeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
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

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

// EmbeddingOp verification
::mlir::LogicalResult mlir::tt::ttnn::EmbeddingOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

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

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

// EmbeddingBackwardOp verification
::mlir::LogicalResult mlir::tt::ttnn::EmbeddingBackwardOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType inputGradType = getInGradient().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // inputType checks:
  // 1. Last dimension must be divisible by TILE_WIDTH.
  if (inputType.getShape().back() % TILE_WIDTH != 0) {
    return emitOpError("Input's last dim must be divisible by TILE_WIDTH");
  }

  // weightType must have rank of 2: (dictionary_size, embedding_size).
  if (weightType.getRank() != 2) {
    return emitOpError("Input must be a 2D tensor");
  }

  // inputGradType checks:
  // 1. inputGradType should have rank of 4, first 2 dimensions must be equal
  //    to 1, third dimension should match the volume of inputType, and the
  //    fourth dimension should match the second dimension of weightType.
  // 2. inputGradType must be of type bfloat16 or bfloat8.
  // 3. inputGradType and outputType must have the same dtype.
  if (inputGradType.getRank() != 4) {
    return emitOpError("Input gradient must be a 4D tensor");
  }
  if (inputGradType.getDimSize(0) != 1 || inputGradType.getDimSize(1) != 1) {
    return emitOpError("Input gradient must be in the form (1, 1, R, C)");
  }

  int64_t inputTypeVolume = 1;
  for (int64_t dim : inputType.getShape()) {
    inputTypeVolume *= dim;
  }
  if (inputGradType.getDimSize(2) != inputTypeVolume) {
    return emitOpError("Input gradient first dimension must match the volume "
                       "of the input tensor");
  }
  if (inputGradType.getDimSize(3) != weightType.getDimSize(1)) {
    return emitOpError("Input gradient second dimension must match the second "
                       "dimension of the weight tensor");
  }
  if (inputGradType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input gradient and output must have the same dtype");
  }

  // outputType should have the same shape as weightType.
  if (outputType.getShape() != weightType.getShape()) {
    return emitOpError("Output must have the same shape as weight");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp
//===----------------------------------------------------------------------===//

// Utility methods
static bool isValidDeviceLayout(TensorMemoryLayoutAttr memLayoutAttr) {
  return memLayoutAttr &&
         (memLayoutAttr.getValue() == TensorMemoryLayout::Interleaved ||
          isShardedMemoryLayout(memLayoutAttr.getValue()));
}

// ToMemoryConfigOp verification
::mlir::LogicalResult mlir::tt::ttnn::ToMemoryConfigOp::verify() {
  ::mlir::RankedTensorType inputTy = getInput().getType();
  ::mlir::RankedTensorType outputTy = getResult().getType();
  auto inputLayout =
      mlir::dyn_cast_if_present<TTNNLayoutAttr>(inputTy.getEncoding());
  auto outputLayout =
      mlir::dyn_cast_if_present<TTNNLayoutAttr>(outputTy.getEncoding());
  if (not inputLayout) {
    return emitOpError("Input tensor type missing layout attribute");
  }
  if (not outputLayout) {
    return emitOpError("Output tensor type missing layout attribute");
  }
  BufferType outputBufferType = outputLayout.getBufferType();
  TensorMemoryLayoutAttr outputMemoryLayout = outputLayout.getMemLayout();

  if (isDeviceBufferType(outputBufferType) &&
      !isValidDeviceLayout(outputMemoryLayout)) {
    return emitOpError("Device memory space only supports interleaved or "
                       "sharded memory layouts");
  }

  if (outputBufferType == BufferType::DRAM &&
      outputMemoryLayout.getValue() != TensorMemoryLayout::Interleaved) {
    return emitOpError(
        "Device DRAM memory space only supports interleaved memory layout");
  }

  if (outputLayout.hasShardedTensorMemoryLayout()) {
    if (not outputLayout.hasShardedL1TensorMemoryLayout()) {
      return emitOpError("Sharded tensors layout must reside in L1");
    }
    ::llvm::SmallVector<int64_t> shardShape = outputLayout.getShardShape();
    // Currently TTNN backend only supports 2D shard shape
    if (shardShape.size() != 2) {
      return emitOpError("Shard shape must be 2D");
    }
    if (outputMemoryLayout.getValue() == TensorMemoryLayout::BlockSharded) {
      // TTNN tiles are (32, 32), shard shape must evenly divide the tile shape
      if (shardShape[0] % TILE_HEIGHT != 0 or shardShape[1] % TILE_WIDTH != 0) {
        return emitOpError(
            "Shard shape must divide tile shape (32, 32) evenly");
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

// ToLayoutOp canonicalization
void mlir::tt::ttnn::ToLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // ToLayoutOp can be folded if its input has the same layout as the output of
  // toLayoutOp.
  patterns.add(+[](mlir::tt::ttnn::ToLayoutOp toLayoutOp,
                   mlir::PatternRewriter &rewriter) {
    RankedTensorType previousType = toLayoutOp.getInput().getType();
    TTNNLayoutAttr previousLayout =
        mlir::dyn_cast<TTNNLayoutAttr>(previousType.getEncoding());
    // Verify if input tensor has layout attribute.
    if (!previousLayout) {
      return mlir::failure();
    }

    RankedTensorType currentType = toLayoutOp.getType();
    TTNNLayoutAttr currentLayout =
        mlir::dyn_cast<TTNNLayoutAttr>(currentType.getEncoding());
    // Verify if the output tensor has layout attribute.
    if (!currentLayout) {
      return mlir::failure();
    }

    // Verify that the layouts are the same.
    if (previousLayout != currentLayout) {
      return mlir::failure();
    }

    rewriter.replaceAllUsesWith(toLayoutOp, toLayoutOp->getOperand(0));
    rewriter.eraseOp(toLayoutOp);
    return mlir::success();
  });

  // Two consecutive ToLayoutOps can be merged if the previous op has only one
  // use.
  // df - data format, l - layout, ms - memory
  // space, tml - tensor memory layout
  //
  //                |
  //      -----------------------
  //      |     ToLayoutOp      |                     |
  //      | df1, l1, ms1, tml1  |          -----------------------
  //      -----------------------          |     ToLayoutOp      |
  //                |                 -->  | df2, l1, ms2, tml1  |
  //                |                      -----------------------
  //      -----------------------                     |
  //      |     ToLayoutOp      |
  //      |      df2, ms2       |
  //      -----------------------
  //                |
  //
  patterns.add(+[](mlir::tt::ttnn::ToLayoutOp toLayoutOp,
                   mlir::PatternRewriter &rewriter) {
    // Get the input operand and verify that the previous op is toLayoutOp.
    ToLayoutOp previousToLayoutOp =
        toLayoutOp.getInput().getDefiningOp<ToLayoutOp>();

    // NOLINTNEXTLINE
    if (!previousToLayoutOp) {
      return mlir::failure();
    }

    // Check if the previous op has only one use. We can only merge if the
    // previous op has single use.
    if (!previousToLayoutOp->hasOneUse()) {
      return mlir::failure();
    }

    // Replace the previous op with the merged ToLayoutOp.
    Value mergedToLayout = rewriter.replaceOpWithNewOp<ToLayoutOp>(
        previousToLayoutOp, toLayoutOp.getType(), previousToLayoutOp.getInput(),
        toLayoutOp.getLayoutAttr(),
        toLayoutOp.getDtypeAttr() ? toLayoutOp.getDtypeAttr()
                                  : previousToLayoutOp.getDtypeAttr(),
        toLayoutOp.getMemoryConfigAttr()
            ? toLayoutOp.getMemoryConfigAttr()
            : previousToLayoutOp.getMemoryConfigAttr(),
        toLayoutOp.getDevice());

    // Replace all uses of the current op with the merged ToLayoutOp.
    rewriter.replaceAllUsesWith(toLayoutOp, mergedToLayout);

    // Erase the current op.
    rewriter.eraseOp(toLayoutOp);

    return mlir::success();
  });
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

// LinearOp verification
::mlir::LogicalResult mlir::tt::ttnn::LinearOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  std::optional<::mlir::RankedTensorType> biasType =
      getBias() ? std::make_optional(getBias().getType()) : std::nullopt;
  ::mlir::RankedTensorType outputType = getResult().getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor.
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor.
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimensions for
  // the purpose of the matrix multiplication. After the matrix multiplication,
  // the prepended dimension is removed. Otherwise, check if the LHS needs to be
  // transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards. Otherwise,
  // check if the RHS needs to be transposed.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  } else if (getTransposeB()) {
    std::swap(inputBShape[inputBShape.size() - 1],
              inputBShape[inputBShape.size() - 2]);
  }

  // Verify that the input A and input B has matching inner dimensions.
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A[-1](")
           << inputAShape[inputAShape.size() - 1] << ") and B[-2]("
           << inputBShape[inputBShape.size() - 2]
           << ") must have matching inner dimensions";
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct the
  // expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(inputABatchDims, inputBBatchDims,
                                            broadcastedShape)) {
      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape.
    expectedOutputShape = std::move(broadcastedShape);
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

  if (biasType) {
    // Verify that the input bias is at least 1D tensor.
    if (biasType->getRank() < 1) {
      return emitOpError("Bias must be at least a 1D tensor");
    }

    llvm::SmallVector<int64_t> biasShape(biasType->getShape());

    // Verify that the dimensions of the matmul of A and B are broadcast
    // compatible with input bias.
    llvm::SmallVector<int64_t> matmulShape = expectedOutputShape;
    if (!OpTrait::util::getBroadcastedShape(matmulShape, biasShape,
                                            expectedOutputShape)) {
      return emitOpError("Bias shape(")
             << ttmlir::utils::join(biasShape, ",")
             << ") is not broadcast compatible with the matmul output shape("
             << ttmlir::utils::join(matmulShape, ",") << ")";
    }
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

    return success();
  }

  // Verify that the output shape is correct.
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(")
           << outputShape.size()
           << ") must match the expected output shape rank("
           << expectedOutputShape.size() << ")";
  }

  // Verify each dim of the output shape.
  for (auto [index, outputDim, expectedDim] : llvm::zip(
           llvm::seq(outputShape.size()), outputShape, expectedOutputShape)) {
    if (outputDim != expectedDim) {
      return emitOpError("Output shape dimension[")
             << index << "](" << outputDim
             << ") doesn't match the expected output shape dimension[" << index
             << "](" << expectedDim << ")";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_matmul_ttnn_verify
// MatmulOp verification
::mlir::LogicalResult mlir::tt::ttnn::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  llvm::SmallVector<int64_t> inputAShape(inputAType.getShape());
  llvm::SmallVector<int64_t> inputBShape(inputBType.getShape());

  // Verify that the input A is at least 1D tensor.
  if (inputAType.getRank() < 1) {
    return emitOpError("Input A must be at least a 1D tensor");
  }

  // Verify that the input B is at least 1D tensor.
  if (inputBType.getRank() < 1) {
    return emitOpError("Input B must be at least a 1D tensor");
  }

  // If input A is a vector (1D tensor), 1 is prepended to its dimensions for
  // the purpose of the matrix multiplication. After the matrix multiplication,
  // the prepended dimension is removed. Otherwise, check if the LHS needs to be
  // transposed.
  if (inputAType.getRank() == 1) {
    inputAShape.insert(inputAShape.begin(), 1);
  } else if (getTransposeA()) {
    std::swap(inputAShape[inputAShape.size() - 1],
              inputAShape[inputAShape.size() - 2]);
  }

  // If input B is a vector (1D tensor), a 1 is appended to its dimensions for
  // the purpose of the matrix-vector product and removed afterwards. Otherwise,
  // check if the RHS needs to be transposed.
  if (inputBType.getRank() == 1) {
    inputBShape.push_back(1);
  } else if (getTransposeB()) {
    std::swap(inputBShape[inputBShape.size() - 1],
              inputBShape[inputBShape.size() - 2]);
  }

  // Verify that the input A and input B has matching inner dimensions.
  if (inputAShape[inputAShape.size() - 1] !=
      inputBShape[inputBShape.size() - 2]) {
    return emitOpError("Input A[-1](")
           << inputAShape[inputAShape.size() - 1] << ") and B[-2]("
           << inputBShape[inputBShape.size() - 2]
           << ") must have matching inner dimensions";
  }

  llvm::SmallVector<int64_t> expectedOutputShape;
  // Verify that the batch dimensions are broadcast compatible and construct the
  // expected output shape. If either of input A or input B is at most 2D
  // tensors, the batch dimensions are trivially broadcast compatible.
  if (inputAShape.size() > 2 || inputBShape.size() > 2) {
    llvm::SmallVector<int64_t> inputABatchDims(inputAShape.begin(),
                                               inputAShape.end() - 2);
    llvm::SmallVector<int64_t> inputBBatchDims(inputBShape.begin(),
                                               inputBShape.end() - 2);

    // Verify that the batch dimensions of input A and B are broadcast
    // compatible.
    llvm::SmallVector<int64_t, 4> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(inputABatchDims, inputBBatchDims,
                                            broadcastedShape)) {

      return emitOpError("Batch dimensions of input A(" +
                         ttmlir::utils::join(inputABatchDims, ",") +
                         ") and B(" +
                         ttmlir::utils::join(inputBBatchDims, ",") +
                         ") are not broadcast compatible");
    }

    // Insert the broadcasted batch dimensions in the expected output shape.
    expectedOutputShape = std::move(broadcastedShape);
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

    return success();
  }

  // Verify that the output shape is correct.
  if (outputShape.size() != expectedOutputShape.size()) {
    return emitOpError("Output shape rank(")
           << outputShape.size()
           << ") must match the expected output shape rank("
           << expectedOutputShape.size() << ")";
  }

  // Verify each dim of the output shape.
  for (auto [index, outputDim, expectedDim] : llvm::zip(
           llvm::seq(outputShape.size()), outputShape, expectedOutputShape)) {
    if (outputDim != expectedDim) {
      return emitOpError("Output shape dimension[")
             << index << "](" << outputDim
             << ") doesn't match the expected output shape dimension[" << index
             << "](" << expectedDim << ")";
    }
  }

  return success();
}
// ANCHOR_END: adding_an_op_matmul_ttnn_verify

//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//

// AllocOp verification
::mlir::LogicalResult AllocOp::verify() {
  auto layout = mlir::dyn_cast_if_present<TTNNLayoutAttr>(
      getResult().getType().getEncoding());
  if (not layout) {
    return emitOpError("Result type missing layout attribute");
  }

  if (getSize() == 0) {
    return emitOpError("Alloc size must be non-zero");
  }

  auto bufferType = layout.getBufferType();
  if (bufferType != getBufferType()) {
    return emitOpError(
        "Input tensor layout memory space must match alloc memory space");
  }

  if (isSystemBufferType(getBufferType()) and getAddress() != 0) {
    return emitOpError("Allocating from system memory space must have address "
                       "set to 0, implicitly allocated by the runtime");
  }

  if (isDeviceBufferType(bufferType) and getAddress() == 0) {
    return emitOpError(
        "Allocating from a device memory space must have address "
        "set to a non-zero value, device addresses are statically allocated");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

// RepeatInterleaveOp verification
::mlir::LogicalResult mlir::tt::ttnn::RepeatInterleaveOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();
  uint32_t repeats = getRepeats();
  int32_t dim = getDim();

  // Verify that the input is at least a 1D tensor.
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
  }

  // Check that the repeats is not zero.
  if (repeats == 0) {
    return emitOpError("Repeats attribute must be non-zero");
  }

  // Check that the dim is within the bounds of the input tensor.
  if (dim >= inputType.getRank() || dim < -inputType.getRank()) {
    return emitOpError("Dimension attribute must be within the bounds")
           << "[" << -inputType.getRank() << ", " << inputType.getRank() << ")"
           << ", got " << inputType.getRank();
  }

  // Normalize dim to [0, n) range.
  if (dim < 0) {
    dim += inputType.getRank();
  }

  // Compute the expected output shape.
  llvm::SmallVector<int64_t> expectedOutputShape(inputType.getShape());
  expectedOutputShape[dim] *= repeats;

  // Verify that the output shape matches the expected shape.
  if (outputType.getShape() != ::llvm::ArrayRef(expectedOutputShape)) {
    return emitOpError("Output shape ")
           << "[" << ttmlir::utils::join(outputType.getShape(), ",") << "]"
           << " does not match the expected shape "
           << "[" << ttmlir::utils::join(expectedOutputShape, ",") << "]";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

// SoftmaxOp verification
::mlir::LogicalResult mlir::tt::ttnn::SoftmaxOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

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

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult AllGatherOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t gatherDim = getAllGatherDim();

  if (gatherDim >= inputType.getRank() || gatherDim < -inputType.getRank()) {
    return emitOpError("Invalid gather dimension for all reduce op. Gather "
                       "dimension must be >= to input tensor rank or < -input "
                       "tensor rank, got gather_dim = ")
           << gatherDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ReduceScatterOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t scatterDim = getScatterDim();
  ::mlir::tt::ReduceType reduceType = getReduceType();

  if (scatterDim >= inputType.getRank() || scatterDim < -inputType.getRank()) {
    return emitOpError("Invalid scatter dimension for all reduce op. Scatter "
                       "dimension must be >= to input tensor rank or < -input "
                       "tensor rank, got scatter_dim = ")
           << scatterDim;
  }

  // Currently TTNN only supports the following reduce types. Compiler is able
  // to model the full ReduceType list but only the following can be lowered
  // into TTNN.
  if (reduceType != ::mlir::tt::ReduceType::Sum &&
      reduceType != ::mlir::tt::ReduceType::Max &&
      reduceType != ::mlir::tt::ReduceType::Min) {
    return emitOpError("Invalid reduction op for all reduce op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult AllReduceOp::verify() {
  ::mlir::tt::ReduceType reduceType = getReduceType();

  // Currently TTNN only supports the following reduce types.
  if (reduceType != ::mlir::tt::ReduceType::Sum &&
      reduceType != ::mlir::tt::ReduceType::Max &&
      reduceType != ::mlir::tt::ReduceType::Min) {
    return emitOpError("Invalid reduction op for all reduce op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MeshShardOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult MeshShardOp::verify() {
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  llvm::ArrayRef<int64_t> shardShape = getShardShape();
  ::mlir::tt::MeshShardType shardType = getShardType();

  // Check shard_type is not maximal.
  if (shardType == ::mlir::tt::MeshShardType::Maximal) {
    return emitOpError("Invalid shard_type (maximal) for mesh_shard op.");
  }

  if (shardType == ::mlir::tt::MeshShardType::Devices) {
    // Check if input rank is equal to or greater than two.
    if (inputShape.size() < 2) {
      return emitOpError(
          "Invalid input rank (<2) for mesh_shard op with devices partition.");
    }

    // Check if shardShape is eqaul to or greater than two.
    if (shardShape.size() < 2) {
      return emitOpError(
          "Invalid shard_shape (<2) for mesh_shard op with devices partition.");
    }

    // Check if rank(shardShape) is eqaul to rank(input).
    if (shardShape.size() != inputShape.size()) {
      return emitOpError("Invalid rank(shard_shape) != rank(input) for "
                         "mesh_shard op with devices partition.");
    }

    // Check if overall partition is eqaul to or greater than two.
    int64_t overallPartition = 1;
    for (auto partition : shardShape) {
      // Each partition value is limited to one of the dimensions of hardware
      // mesh. Thus, overallPartition remains lower than or equal to the number
      // of multi-chips.
      overallPartition *= partition;
    }
    if (overallPartition < 2) {
      return emitOpError("Invalid overall partition (<2) for mesh_shard op "
                         "with devices partition.");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult UpdateCacheOp::verify() {
  if (getBatchOffset() != 0) {
    return emitOpError(
        "Only single-batch is supported. Batch offset must be 0");
  }

  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const DataType cacheDataType =
      elementTypeToDataType(cacheType.getElementType());
  const DataType inputDataType =
      elementTypeToDataType(inputType.getElementType());

  if (cacheDataType != inputDataType) {
    return emitOpError(
        "Cache and input tensors must have the same dtype. "
        "Got cache dtype = " +
        DataTypeEnumToString(cacheDataType) +
        ", input dtype = " + DataTypeEnumToString(inputDataType));
  }

  if (cacheType.getRank() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (inputType.getShape()[2] != 1) {
    return emitOpError("Input tensor requires that dim 2 have size 1, got "
                       "input dim 2 size = " +
                       std::to_string(inputType.getShape()[2]));
  }

  if (cacheType.getShape()[0] != inputType.getShape()[0] ||
      cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "all dimensions except dim 2. Got cache shape (" +
                       std::to_string(cacheType.getShape()[0]) + ", " +
                       std::to_string(cacheType.getShape()[1]) + ", " +
                       std::to_string(cacheType.getShape()[2]) + ", " +
                       std::to_string(cacheType.getShape()[3]) +
                       "), input shape ()" +
                       std::to_string(inputType.getShape()[0]) + "x" +
                       std::to_string(inputType.getShape()[1]) + "x" +
                       std::to_string(inputType.getShape()[2]) + "x" +
                       std::to_string(inputType.getShape()[3]) + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult FillCacheOp::verify() {
  if (getBatchOffset() != 0) {
    return emitOpError(
        "Only single-batch is supported. Batch offset must be 0");
  }

  const ::mlir::RankedTensorType cacheType = getCache().getType();
  const ::mlir::RankedTensorType inputType = getInput().getType();

  const DataType cacheDataType =
      elementTypeToDataType(cacheType.getElementType());
  const DataType inputDataType =
      elementTypeToDataType(inputType.getElementType());

  if (cacheDataType != inputDataType) {
    return emitOpError(
        "Cache and input tensors must have the same dtype. "
        "Got cache dtype = " +
        DataTypeEnumToString(cacheDataType) +
        ", input dtype = " + DataTypeEnumToString(inputDataType));
  }

  if (cacheType.getRank() != 4) {
    return emitOpError("Cache tensor must be a 4D tensor");
  }

  if (inputType.getRank() != 4) {
    return emitOpError("Input tensor must be a 4D tensor");
  }

  if (inputType.getShape()[2] > cacheType.getShape()[2]) {
    return emitOpError(
        "Input tensor requires that dim 2 have a size which is less than or "
        "equal to the size of dim 2 of the cache tensor. Got cache dim 2 size "
        "= " +
        std::to_string(cacheType.getShape()[2]) +
        ", input dim 2 size = " + std::to_string(inputType.getShape()[2]));
  }

  if (cacheType.getShape()[0] != inputType.getShape()[0] ||
      cacheType.getShape()[1] != inputType.getShape()[1] ||
      cacheType.getShape()[3] != inputType.getShape()[3]) {
    return emitOpError("Cache tensor shape must match input tensor shape on "
                       "all dimensions except dim 2. Got cache shape (" +
                       std::to_string(cacheType.getShape()[0]) + ", " +
                       std::to_string(cacheType.getShape()[1]) + ", " +
                       std::to_string(cacheType.getShape()[2]) + ", " +
                       std::to_string(cacheType.getShape()[3]) +
                       "), input shape (" +
                       std::to_string(inputType.getShape()[0]) + ", " +
                       std::to_string(inputType.getShape()[1]) + ", " +
                       std::to_string(inputType.getShape()[2]) + ", " +
                       std::to_string(inputType.getShape()[3]) + ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

// PermuteOp verification
::mlir::LogicalResult mlir::tt::ttnn::PermuteOp::verify() {
  llvm::ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  const size_t inputRank = inputShape.size();
  llvm::ArrayRef<int64_t> resultShape = getResult().getType().getShape();

  // Check that given attribute `permutation` is a valid permutation of the
  // dimensions.
  llvm::ArrayRef<int64_t> permutation = getPermutation();
  llvm::SmallVector<int64_t> dimensions(inputRank);
  std::iota(dimensions.begin(), dimensions.end(), 0);
  if (inputRank != permutation.size() ||
      !std::is_permutation(permutation.begin(), permutation.end(),
                           dimensions.begin())) {
    return emitOpError("Expected a permutation of (")
           << ttmlir::utils::join(dimensions, ", ")
           << "), got (" + ttmlir::utils::join(permutation, ", ") << ")";
  }

  // Check that the result shape matches the shape of input tensor after
  // permutation is applied.
  llvm::SmallVector<int64_t> expectedResultShape =
      ttmlir::utils::applyPermutation(inputShape, permutation);
  if (!llvm::equal(expectedResultShape, resultShape)) {
    return emitOpError("Expected result shape (" +
                       ttmlir::utils::join(expectedResultShape, ", ") +
                       "), got (" + ttmlir::utils::join(resultShape, ", ") +
                       ")");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

// UpsampleOp verification
::mlir::LogicalResult UpsampleOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getResult().getType();

  // Input tensor is assumed to be 4D tensor.
  if (inputType.getRank() != 4) {
    return emitOpError("Expected rank of input tensor is 4, got rank " +
                       std::to_string(inputType.getRank()));
  }
  if (outputType.getRank() != 4) {
    return emitOpError("Expected rank of output tensor is 4, got rank " +
                       std::to_string(outputType.getRank()));
  }

  auto scaleFactor = ttmlir::utils::getPairOfInteger<int32_t>(getScaleFactor());
  if (auto error = scaleFactor.takeError()) {
    return emitOpError() << llvm::toString(std::move(error));
  }
  int32_t scaleH = scaleFactor->first;
  int32_t scaleW = scaleFactor->second;

  if (scaleH <= 0 || scaleW <= 0) {
    return emitOpError("Scale factors H = ")
           << scaleH << " and W = " << scaleW << " must be positive integers";
  }

  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  // Input tensor is assumed to be in NHWC format.
  enum Dimensions { DIM_N = 0, DIM_H = 1, DIM_W = 2, DIM_C = 3 };
  if (inputShape[DIM_H] * scaleH != outputShape[DIM_H]) {
    return emitOpError("Expected output H dimension to be input H dimension * "
                       "scaleH = ")
           << (inputShape[DIM_H] * scaleH) << ", got " << outputShape[DIM_H];
  }
  if (inputShape[DIM_W] * scaleW != outputShape[DIM_W]) {
    return emitOpError("Expected output W dimension to be input W dimension * "
                       "scaleW = ")
           << (inputShape[DIM_W] * scaleW) << ", got " << outputShape[DIM_W];
  }
  if (inputShape[DIM_N] != outputShape[DIM_N]) {
    return emitOpError("Expected output N dimension to be ")
           << inputShape[DIM_N] << ", got " << outputShape[DIM_N];
  }
  if (inputShape[DIM_C] != outputShape[DIM_C]) {
    return emitOpError("Expected output C dimension to be ")
           << inputShape[DIM_C] << ", got " << outputShape[DIM_C];
  }

  // Verify that the mode attribute is one of the legal modes. These two modes
  // are currently only supported modes in TTNN.
  llvm::SmallVector<llvm::StringRef> legalModes = {"nearest", "bilinear"};
  if (std::find(legalModes.begin(), legalModes.end(), getMode()) ==
      legalModes.end()) {
    return emitOpError("Expected modes are (")
           << llvm::join(legalModes, ", ") << "), got \"" << getMode() << "\"";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

// Common verifier for all Reduction ops.
static mlir::LogicalResult
verifyReduceOp(mlir::Operation *reduceOp, mlir::RankedTensorType inputType,
               const std::optional<mlir::ArrayAttr> &reduceDims, bool keepDim,
               ::llvm::ArrayRef<int64_t> specifiedOutputShape) {
  if (!reduceDims) {
    return mlir::success();
  }

  int64_t inputTensorRank = inputType.getRank();

  // Calculate output shape for given args.
  //
  llvm::SmallVector<int64_t> calculatedOutputShape;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    bool isDimInReduceDims =
        llvm::any_of(*reduceDims, [i, inputTensorRank](mlir::Attribute attr) {
          int64_t reduceDim = mlir::cast<mlir::IntegerAttr>(attr).getInt();
          // Check for match even if negative dim is used.
          //
          return reduceDim == i || (reduceDim + inputTensorRank) == i;
        });

    // If dim is being reduced on, the dim will have size of 1 if keepDim==true,
    // otherwise the dim is erased.
    //
    if (!isDimInReduceDims) {
      calculatedOutputShape.push_back(inputType.getDimSize(i));
    } else if (keepDim) {
      calculatedOutputShape.push_back(1);
    }
  }

  // Cover edge case where all dims are reduced, and keepDim==false.
  if (calculatedOutputShape.size() == 0 && keepDim == false) {
    calculatedOutputShape.push_back(1);
  }

  // Finally, compare shapes.
  //
  if (!llvm::equal(specifiedOutputShape, calculatedOutputShape)) {
    return reduceOp->emitOpError(
        "Expected output shape (" +
        ttmlir::utils::join(specifiedOutputShape, ", ") + "), got (" +
        ttmlir::utils::join(calculatedOutputShape, ", ") + ")");
  }

  return mlir::success();
}

// Verifier for Reduce ProdOp.
static mlir::LogicalResult verifyReduceProdOp(mlir::Operation *reduceOp,
                                              mlir::RankedTensorType inputType,
                                              bool allDimensions) {
  int64_t inputTensorRank = inputType.getRank();
  mlir::Type elementType = inputType.getElementType();

  if (inputTensorRank > 4) {
    return reduceOp->emitOpError(
        "Input tensor rank is greater than 4 for reduce(product).");
  }
  // [TODO](mmanzoor) Add workaround to typecast the input tensor to bfloat16
  // then typecast the output again to match the requirements.
  // https://github.com/tenstorrent/tt-mlir/issues/1864
  if (allDimensions && !elementType.isBF16()) {
    return reduceOp->emitOpError("TTNN only supports Reduce(prod) along all "
                                 "dimensions for bfloat16 datatype.");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

// MaxOp verification.
::mlir::LogicalResult MaxOp::verify() {
  return verifyReduceOp(getOperation(), getInput().getType(), getDimArg(),
                        getKeepDim(), getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// MeanOp verification.
::mlir::LogicalResult MeanOp::verify() {
  return verifyReduceOp(getOperation(), getInput().getType(), getDimArg(),
                        getKeepDim(), getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

// SumOp verification.
::mlir::LogicalResult SumOp::verify() {
  return verifyReduceOp(getOperation(), getInput().getType(), getDimArg(),
                        getKeepDim(), getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce MinOp
//===----------------------------------------------------------------------===//

// MinOp verification.
::mlir::LogicalResult MinOp::verify() {
  return verifyReduceOp(getOperation(), getInput().getType(), getDimArg(),
                        getKeepDim(), getResult().getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ProdOp
//===----------------------------------------------------------------------===//

// ProdOp verification.
::mlir::LogicalResult ProdOp::verify() {
  return verifyReduceProdOp(getOperation(), getInput().getType(),
                            getAllDimensions());
}

} // namespace mlir::tt::ttnn
