// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.cpp.inc"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <numeric>
#include <string>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// BitwiseXorOp
//===----------------------------------------------------------------------===//

// BitwiseXorOp canonicalization
void mlir::tt::ttir::BitwiseXorOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // x ^ x == 0
  patterns.add(
      +[](mlir::tt::ttir::BitwiseXorOp op, mlir::PatternRewriter &rewriter) {
        if (op.getInputs()[0] != op.getInputs()[1]) {
          return mlir::failure();
        }

        mlir::RankedTensorType tensorType =
            mlir::cast<mlir::RankedTensorType>(op.getInputs()[0].getType());
        auto elementType = tensorType.getElementType();
        Attribute zeroAttr;
        if (mlir::isa<mlir::FloatType>(elementType)) {
          zeroAttr = mlir::FloatAttr::get(elementType, 0.0);
        } else if (mlir::isa<mlir::IntegerType>(elementType)) {
          zeroAttr = mlir::IntegerAttr::get(elementType, 0);
        } else {
          return mlir::failure();
        }
        auto resultType = mlir::SplatElementsAttr::get(tensorType, zeroAttr);

        rewriter.replaceOpWithNewOp<ttir::ConstantOp>(
            op, op->getOperand(0).getType(), resultType);
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// BroadcastOp folder
::mlir::OpFoldResult mlir::tt::ttir::BroadcastOp::fold(FoldAdaptor adaptor) {
  // If the input doesn't change the shape, we can fold the operation.
  if (llvm::all_of(getBroadcastDimensions(),
                   [](const int32_t dim) { return dim == 1; })) {
    return getInput();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ClampOp::verify() {
  const RankedTensorType inputTensorType =
      mlir::cast<RankedTensorType>(getInput().getType());

  const RankedTensorType outputTensorType =
      mlir::cast<RankedTensorType>(getResult().getType());

  if (inputTensorType != outputTensorType) {
    return emitOpError("input and output must have same shape.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ArangeOp::verify() {
  int64_t start = getStart();
  int64_t end = getEnd();
  int64_t step = getStep();

  if (step == 0) {
    return emitOpError("Step value cannot be zero");
  }

  int64_t numValues = (end - start) / step;

  if (numValues <= 0) {
    return emitOpError() << "Invalid range: start=" << start << ", end=" << end
                         << ", step=" << step;
  }

  if (numValues != getType().getDimSize(getArangeDimension())) {
    return emitOpError() << "Output tensor shape must be " << numValues
                         << " at dim " << getArangeDimension()
                         << " (since start=" << start << ", end=" << end
                         << ", step=" << step << "), but got "
                         << getType().getDimSize(getArangeDimension());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// ConstantOp folder
::mlir::OpFoldResult mlir::tt::ttir::ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

// GetDimensionSizeOp verification
::mlir::LogicalResult mlir::tt::ttir::GetDimensionSizeOp::verify() {
  RankedTensorType inputTensorType = getOperand().getType();

  int64_t dimensionIndex = getDimension();

  if (dimensionIndex >=
      static_cast<int64_t>(inputTensorType.getShape().size())) {
    return failure();
  };

  return success();
}

// GetDimensionSizeOp folder
::mlir::OpFoldResult
mlir::tt::ttir::GetDimensionSizeOp::fold(FoldAdaptor adaptor) {
  RankedTensorType inputTensorType = getOperand().getType();
  uint32_t dimensionIndex = getDimension();
  uint32_t dimSize = inputTensorType.getShape()[dimensionIndex];

  return mlir::DenseElementsAttr::get<uint32_t>(getType(), dimSize);
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

// Conv2dOp verification
::mlir::LogicalResult mlir::tt::ttir::Conv2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getOutput().getType();
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

  constexpr unsigned int BATCH_DIM = 0, HEIGHT_DIM = 1, WIDTH_DIM = 2,
                         CHANNEL_DIM = 3;
  if (!getFlattenedCompatInfo().has_value() &&
      inputType.getDimSize(BATCH_DIM) != outputType.getDimSize(BATCH_DIM)) {
    return emitOpError()
           << "Batch size from the input tensor ("
           << inputType.getDimSize(BATCH_DIM)
           << ") must match the first dimension of the output tensor ("
           << outputType.getDimSize(BATCH_DIM) << ")";
  }

  uint32_t inputHeight = inputType.getDimSize(HEIGHT_DIM);
  uint32_t inputWidth = inputType.getDimSize(WIDTH_DIM);
  uint32_t inChannels = inputType.getDimSize(CHANNEL_DIM);
  uint32_t outChannels = outputType.getDimSize(CHANNEL_DIM);
  if (getFlattenedCompatInfo().has_value()) {
    inputHeight = getFlattenedCompatInfo()->getInputHeight();
    inputWidth = getFlattenedCompatInfo()->getInputWidth();
    inChannels = getFlattenedCompatInfo()->getInChannels();
    outChannels = getFlattenedCompatInfo()->getOutChannels();
  }

  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(getStride());
  if (auto error = stride.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for stride";
  }
  if (stride->first < 1 || stride->second < 1) {
    return emitOpError("Stride attribute values must be greater than 0");
  }

  auto padding = ttmlir::utils::getQuadrupleOfInteger<int32_t>(getPadding());
  if (auto error = padding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for padding";
  }

  auto [paddingTop, paddingLeft, paddingBottom, paddingRight] = *padding;
  if (paddingTop < 0 || paddingBottom < 0 || paddingLeft < 0 ||
      paddingRight < 0) {
    return emitOpError(
        "Padding attribute values must be greater than or equal to 0");
  }
  int32_t verticalPadding = paddingTop + paddingBottom;
  int32_t horizontalPadding = paddingLeft + paddingRight;

  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(getDilation());
  if (auto error = dilation.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for dilation";
  }
  if (dilation->first < 1 || dilation->second < 1) {
    return emitOpError("Dilation attribute values must be greater than 0");
  }

  constexpr unsigned int WEIGHT_OUT_CHANNEL_DIM = 0, WEIGHT_IN_CHANNEL_DIM = 1;
  constexpr unsigned int WEIGHT_KERNEL_HEIGHT_DIM = 2,
                         WEIGHT_KERNEL_WIDTH_DIM = 3;
  llvm::SmallVector<int64_t> kernelSize{
      static_cast<int32_t>(weightType.getDimSize(WEIGHT_KERNEL_HEIGHT_DIM)),
      static_cast<int32_t>(weightType.getDimSize(WEIGHT_KERNEL_WIDTH_DIM))};

  llvm::SmallVector<uint32_t, 2> paddedInputSize{
      inputHeight + verticalPadding, inputWidth + horizontalPadding};
  llvm::SmallVector<uint32_t, 2> effectiveKernelSize{
      static_cast<uint32_t>(kernelSize[0] +
                            (kernelSize[0] - 1) * (dilation->first - 1)),
      static_cast<uint32_t>(kernelSize[1] +
                            (kernelSize[1] - 1) * (dilation->second - 1))};
  if (paddedInputSize[0] < effectiveKernelSize[0] ||
      paddedInputSize[1] < effectiveKernelSize[1]) {
    return emitOpError()
           << "Calculated padded input size per channel: ("
           << paddedInputSize[0] << " x " << paddedInputSize[1]
           << "). Kernel size: (" << effectiveKernelSize[0] << " x "
           << effectiveKernelSize[1]
           << "). Kernel size can't be greater than actual input size";
  }

  uint32_t groups = getGroups();
  if (inChannels % groups != 0) {
    return emitOpError() << "Number of input channels from input tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << inChannels << " input channels and "
                         << groups << " groups";
  }

  if (outChannels % groups != 0) {
    return emitOpError() << "Number of output channels from output tensor must "
                            "be divisible by the number of groups. "
                         << "Got " << outChannels << " output channels and "
                         << groups << " groups";
  }

  llvm::ArrayRef<std::int64_t> weightShape = weightType.getShape();
  if (outChannels != weightShape[WEIGHT_OUT_CHANNEL_DIM]) {
    return emitOpError() << "Number of output channels from output tensor must "
                            "match the first dimension of the weight tensor. "
                         << "Got " << outChannels << " output channels and "
                         << weightShape[WEIGHT_OUT_CHANNEL_DIM]
                         << " in the weight tensor";
  }

  if (inChannels / groups != weightShape[WEIGHT_IN_CHANNEL_DIM]) {
    return emitOpError() << "Number of input channels per group must match "
                            "the second dimension of the weight tensor. "
                         << "Got " << (inChannels / groups)
                         << " input channels per group and "
                         << weightShape[WEIGHT_IN_CHANNEL_DIM]
                         << " in the weight tensor";
  }

  if (bias && bias->getDimSize(CHANNEL_DIM) != outChannels) {
    return emitOpError() << "Mismatch in bias tensor dimensions. "
                         << "Bias tensor has " << bias->getDimSize(CHANNEL_DIM)
                         << " channels, "
                         << "but the output tensor has " << outChannels
                         << " channels";
  }

  int32_t calculatedHOut = (inputHeight + verticalPadding -
                            dilation->first * (kernelSize[0] - 1) - 1) /
                               stride->first +
                           1;
  int32_t calculatedWOut = (inputWidth + horizontalPadding -
                            dilation->second * (kernelSize[1] - 1) - 1) /
                               stride->second +
                           1;
  if (!getFlattenedCompatInfo().has_value()) {
    if (calculatedHOut != outputType.getDimSize(HEIGHT_DIM) ||
        calculatedWOut != outputType.getDimSize(WIDTH_DIM)) {
      return emitOpError()
             << "Mismatch between calculated and got output height and width. "
             << "Calculated: (" << calculatedHOut << " x " << calculatedWOut
             << "). "
             << "Got output tensor height and width: ("
             << outputType.getDimSize(HEIGHT_DIM) << " x "
             << outputType.getDimSize(WIDTH_DIM) << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

// ConvTranspose2dOp verification
mlir::LogicalResult mlir::tt::ttir::ConvTranspose2dOp::verify() {
  mlir::RankedTensorType inputType = getInput().getType();
  mlir::RankedTensorType weightType = getWeight().getType();
  mlir::RankedTensorType outputType = getOutput().getType();
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

  if (inputType.getShape()[0] != outputType.getShape()[0]) {
    return emitOpError("Batch size of input and output tensors must match");
  }

  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(getStride());
  if (auto error = stride.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for stride";
  }
  if (stride->first < 1 || stride->second < 1) {
    return emitOpError("Stride values must be greater than 0");
  }

  auto padding = ttmlir::utils::getQuadrupleOfInteger<int32_t>(getPadding());
  if (auto error = padding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for padding";
  }

  auto [paddingTop, paddingLeft, paddingBottom, paddingRight] = *padding;
  if (paddingTop < 0 || paddingBottom < 0 || paddingLeft < 0 ||
      paddingRight < 0) {
    return emitOpError("Padding values must be greater or equal than 0");
  }
  int32_t verticalPadding = paddingTop + paddingBottom;
  int32_t horizontalPadding = paddingLeft + paddingRight;

  auto outputPadding =
      ttmlir::utils::getPairOfInteger<int32_t>(getOutputPadding());
  if (auto error = outputPadding.takeError()) {
    return emitOpError() << llvm::toString(std::move(error))
                         << " for output padding";
  }
  if (outputPadding->first < 0 || outputPadding->second < 0) {
    return emitOpError("Output padding values must be greater or equal than 0");
  }

  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(getDilation());
  if (auto error = dilation.takeError()) {
    return emitOpError() << llvm::toString(std::move(error)) << " for dilation";
  }
  if (dilation->first < 1 || dilation->second < 1) {
    return emitOpError("Dilation values must be greater than 0");
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

  int32_t expectedHOut = (Hin - 1) * stride->first - verticalPadding +
                         dilation->first * (kernelHeight - 1) +
                         outputPadding->first + 1;
  int32_t expectedWOut = (Win - 1) * stride->second - horizontalPadding +
                         dilation->second * (kernelWidth - 1) +
                         outputPadding->second + 1;
  if (expectedHOut < 0 || expectedWOut < 0) {
    return emitOpError() << "Given input size per channel: (" << Hin << " x "
                         << Win << "). "
                         << "Calculated output size per channel: ("
                         << expectedHOut << " x " << expectedWOut << "). "
                         << "Output size is too small";
  }

  int32_t HOut = outputType.getDimSize(outputType.getRank() - 3);
  int32_t WOut = outputType.getDimSize(outputType.getRank() - 2);
  if (HOut != expectedHOut || WOut != expectedWOut) {
    return emitOpError() << "Mismatch between expected output size per channel "
                            "and got output tensor dimensions. "
                         << "Expected: (" << expectedHOut << " x "
                         << expectedWOut << "), "
                         << "got: (" << HOut << " x " << WOut << ").";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ConvolutionOp::verify() {
  if (getConvolutionLayout().getInputSpatialDimensions().size() !=
      getConvolutionLayout().getOutputSpatialDimensions().size()) {
    return emitOpError("Convolution input, output, and kernel must have the "
                       "same number of spatial dimensions");
  }
  if (getConvolutionLayout().getInputSpatialDimensions().size() !=
      getConvolutionLayout().getKernelSpatialDimensions().size()) {
    return emitOpError("Convolution input, output, and kernel must have the "
                       "same number of spatial dimensions");
  }

  // Subtract 2 from the rank as to not count batch and feature dimension
  if (getInput().getType().getRank() - 2 !=
      static_cast<int64_t>(
          getConvolutionLayout().getInputSpatialDimensions().size())) {
    return emitOpError("Input tensor must have the same number of spatial "
                       "dimensions as specified in the ConvolutionLayout");
  }

  if (getWeight().getType().getRank() - 2 !=
      static_cast<int64_t>(
          getConvolutionLayout().getKernelSpatialDimensions().size())) {
    return emitOpError("Weight tensor must have the same number of spatial "
                       "dimensions as specified in the ConvolutionLayout");
  }

  std::optional<::mlir::RankedTensorType> biasType =
      getBias().getImpl() ? std::make_optional(getBias().getType())
                          : std::nullopt;

  if (biasType.has_value()) {
    if (biasType->getRank() != 4) {
      return emitOpError("Bias must be a 4D tensor");
    }
  }

  if (getWindowStrides().size() !=
      getConvolutionLayout().getInputSpatialDimensions().size()) {
    return emitOpError("Window strides must have the same number of elements "
                       "as the spatial dimensions of the input tensor");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// PoolingOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::PoolingOp::verify() {

  uint32_t inputRank =
      mlir::cast<RankedTensorType>(getInputs()[0].getType()).getRank();

  for (auto input : getInputs()) {
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    if (inputType.getRank() != inputRank) {
      return emitOpError("All input tensors must have the same rank");
    }
  }

  if (getWindowStrides().size() != inputRank) {
    return emitOpError("Window strides must have the same number of elements "
                       "as the rank of the input tensor");
  }

  if (getWindowDilations().size() != inputRank) {
    return emitOpError("Window dilations must have the same number of elements "
                       "as the rank of the input tensor");
  }

  if (getWindowDimensions().size() != inputRank) {
    return emitOpError(
        "Window dimensions must have the same number of elements "
        "as the rank of the input tensor");
  }

  if (getPadding().size() != 2 * inputRank) {
    return emitOpError("Padding must have the same number of elements as twice "
                       "the rank of the input tensor");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MaxPool2dOp
//===----------------------------------------------------------------------===//

// MaxPool2dOp verification
::mlir::LogicalResult mlir::tt::ttir::MaxPool2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  std::vector<int64_t> inputShape = inputType.getShape().vec();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  std::vector<int64_t> outputShape = outputType.getShape().vec();

  if (inputType.getRank() != 4) {
    return emitOpError()
           << "Input tensor rank must be 4. Recieved input with rank "
           << inputType.getRank() << ". Shape: (" << inputShape << ").";
  }

  if (getFlattenedCompatInfo().has_value()) {
    inputShape[0] = getFlattenedCompatInfo()->getBatchSize();
    inputShape[1] = getFlattenedCompatInfo()->getInputHeight();
    inputShape[2] = getFlattenedCompatInfo()->getInputWidth();
    inputShape[3] = getFlattenedCompatInfo()->getInChannels();

    if (getFlattenedCompatInfo()->getInChannels() !=
        getFlattenedCompatInfo()->getOutChannels()) {
      return emitOpError() << "In channels ("
                           << getFlattenedCompatInfo()->getInChannels()
                           << ") must match out channels ("
                           << getFlattenedCompatInfo()->getOutChannels()
                           << ") as defined in flattened_compat_info.";
    }
  }

  if (outputShape.back() != inputShape.back()) {
    return emitOpError() << "Output tensor channels (" << outputShape.back()
                         << ") must match input tensor channels ("
                         << inputShape.back() << ").";
  }

  if (getKernelHeight() > inputShape[1]) {
    return emitOpError() << "Kernel height " << getKernelHeight()
                         << " is greater than input height " << inputShape[1]
                         << (getFlattenedCompatInfo().has_value()
                                 ? "(as defined in flattened_compat_info)"
                                 : "")
                         << ". This MaxPool2d configuration is invalid.";
  }

  if (getKernelWidth() > inputShape[2]) {
    return emitOpError() << "Kernel width " << getKernelWidth()
                         << " is greater than input width " << inputShape[2]
                         << (getFlattenedCompatInfo().has_value()
                                 ? "(as defined in flattened_compat_info)"
                                 : "")
                         << ". This MaxPool2d configuration is invalid.";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// ConcatOp verification
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

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// PadOp verification
::mlir::LogicalResult mlir::tt::ttir::PadOp::verify() {

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
// ReshapeOp
//===----------------------------------------------------------------------===//

// ReshapeOp verification
::mlir::LogicalResult mlir::tt::ttir::ReshapeOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  auto shape = getShape();
  int64_t shapeSize = static_cast<int64_t>(shape.size());

  // Check that the shape attribute is non-empty.
  if (shapeSize == 0) {
    return emitOpError("Shape attribute must be non-empty");
  }

  // Check that the shape size matches the rank of the output tensor.
  if (shapeSize != static_cast<int64_t>(outputType.getRank())) {
    return emitOpError() << "Shape attribute size " << shapeSize
                         << " must match output tensor rank "
                         << outputType.getRank();
  }

  // Cardinality of the input and output tensors must be the same.
  if (inputType.getNumElements() != outputType.getNumElements()) {
    return emitOpError() << "Input tensor number of elements "
                         << inputType.getNumElements()
                         << " and output tensor number of elements "
                         << outputType.getNumElements() << " must be the same";
  }

  bool hasNegative = false;
  auto outputShape = outputType.getShape();

  // Check that all dimensions are positive except for at most one -1.
  // Check that the non-negative dimensions match the output tensor shape.
  // Calculate the product of the known dimensions.
  for (int64_t i = 0; i < shapeSize; i++) {
    int64_t dimValue = mlir::cast<IntegerAttr>(shape[i]).getInt();

    if (dimValue == -1) {
      if (hasNegative) {
        return emitOpError("Shape attribute must have at most one -1 element");
      }
      hasNegative = true;
    } else {
      if (dimValue <= 0) {
        return emitOpError(
            "All dimensions must be positive except the one with -1");
      }

      // Ensure that the non-negative dimensions match the output tensor shape.
      if (dimValue != outputShape[i]) {
        return emitOpError()
               << "Shape attribute " << dimValue
               << " must match the output tensor shape " << outputShape[i]
               << " at index " << i << " for dimension that is not -1";
      }
    }
  }

  return success();
}

mlir::OpFoldResult foldIdentityReshape(mlir::tt::ttir::ReshapeOp op) {
  // Fold the operation if the type of the input and output types are the same.
  if (op.getType() == op.getInput().getType()) {
    return op.getInput();
  }
  return nullptr;
}

mlir::OpFoldResult foldOperandReshape(mlir::tt::ttir::ReshapeOp op) {
  // Back to back reshapes can be replaced with the final reshape.
  if (auto reshapeOperand =
          op.getInput().getDefiningOp<mlir::tt::ttir::ReshapeOp>()) {
    op.setOperand(0, reshapeOperand.getInput());
    return op.getResult();
  }
  return nullptr;
}

// ReshapeOp folder
::mlir::OpFoldResult mlir::tt::ttir::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityReshape(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldOperandReshape(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// BroadcastOp verification
::mlir::LogicalResult mlir::tt::ttir::BroadcastOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

  // Sanity check to make sure that input rank matches the rank of the output
  // tensor.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank of " << inputType.getRank()
                         << " does not match output tensor rank of "
                         << outputType.getRank();
  }

  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  // Verify that inputShape can be legally broadcasted to outputShape.
  llvm::SmallVector<int64_t> broadcastedShape;
  if (!mlir::OpTrait::util::getBroadcastedShape(inputShape, outputShape,
                                                broadcastedShape)) {
    return emitOpError() << "Input tensor shape ("
                         << ttmlir::utils::join(inputShape, ",")
                         << ") is not broadcastable to output shape ("
                         << ttmlir::utils::join(outputShape, ",") << ")";
  }

  auto broadcastDimensions = getBroadcastDimensions();

  // Check that the shape size matches the rank of the output tensor.
  if (static_cast<int64_t>(broadcastDimensions.size()) != inputType.getRank()) {
    return emitOpError("Input tensor rank should match output tensor rank.");
  }

  // Verify that each dimension of the inputShape multiplied by corresponding
  // broadcast dimension is equal to the outputShape dimension.
  for (size_t i = 0; i < broadcastDimensions.size(); i++) {
    int64_t dimValue = broadcastDimensions[i];
    if (inputShape[i] * dimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",") << ") index "
                           << i << " does not broadcast to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using broadcast value " << dimValue;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// SliceOp verification
::mlir::LogicalResult mlir::tt::ttir::SliceOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::ArrayAttr begins = getBeginsAttr();
  ::mlir::ArrayAttr ends = getEndsAttr();
  ::mlir::ArrayAttr stepAttr = getStepAttr();
  ::mlir::RankedTensorType outputType = getOutput().getType();

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
// IndexOp
//===----------------------------------------------------------------------===//

// ANCHOR: decomposing_an_op_index_ttir_verify
// IndexOp verification
::mlir::LogicalResult mlir::tt::ttir::IndexOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  int32_t dim = getDim();
  int32_t begin = getBegin();
  int32_t end = getEnd();
  int32_t step = getStep();

  // Verify that the input is at least 1D tensor
  if (inputType.getRank() < 1) {
    return emitOpError("Input must be at least a 1D tensor");
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

  // Verify that the dim attribute is within the bounds of the input tensor
  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension index " << dim
                         << ". Input tensor rank is " << inputType.getRank();
  }

  // Verify begin, end, step and the output tensor dimensions
  int64_t dimSize = inputShape[dim];

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
                         << std::to_string(dim) << ". Expected value in range ["
                         << std::to_string(-dimSize) << ", " << dimSize
                         << "), got " << begin
                         << ". Input shape: " << inputShapeStr;
  }
  if (adjustedEnd < 0 || adjustedEnd > dimSize) {
    return emitOpError() << "Invalid end index for dimension "
                         << std::to_string(dim) << ". Expected value in range ["
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
    return emitOpError("Step value for dimension " + std::to_string(dim) +
                       " cannot be zero");
  }

  if (step > 0 && adjustedBegin > adjustedEnd) {
    return emitOpError() << "For positive step, begin index must be less "
                            "than or equal to end index for dimension "
                         << dim << ". Got begin: " << beginValueMessage
                         << ", end: " << endValueMessage << ", step: " << step
                         << ", input shape: " << inputShapeStr;
  }

  if (step < 0 && adjustedBegin < adjustedEnd) {
    return emitOpError() << "For negative step, begin index must be greater "
                            "than or equal to end index for dimension "
                         << dim << ". Got begin: " << beginValueMessage
                         << ", end: " << endValueMessage << ", step: " << step
                         << ", input shape: " << inputShapeStr;
  }

  // Calculate the expected size of the output dimension
  int32_t expectedDimSize =
      (std::abs(adjustedEnd - adjustedBegin) + std::abs(step) - 1) /
      std::abs(step);
  if (outputType.getDimSize(dim) != expectedDimSize) {
    return emitOpError() << "Mismatch in dimension " << std::to_string(dim)
                         << " of the output tensor: expected size "
                         << expectedDimSize << ", but got "
                         << outputType.getDimSize(dim);
  }

  return success();
}
// ANCHOR_END: decomposing_an_op_index_ttir_verify

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// SelectOp verification
::mlir::LogicalResult mlir::tt::ttir::SelectOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError("Input and output tensors must have the same rank.");
  }

  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("Input and output tensors must have the same element "
                       "type.");
  }

  int32_t dim = getDim();
  int32_t origDim = dim;
  if (dim < 0) {
    dim += inputType.getRank();
  }

  if (dim < 0 || dim >= inputType.getRank()) {
    return emitOpError() << "Invalid dimension " << origDim
                         << " for select op with input tensor rank "
                         << inputType.getRank();
  }

  int32_t dimSize = inputType.getDimSize(dim);

  int32_t stride = getStride();
  if (stride == 0) {
    stride = dimSize;
  }

  if (stride < 0) {
    return emitOpError() << "Invalid stride " << stride << " for dimension "
                         << dim << ", stride must be non-negative";
  }

  if (stride > dimSize) {
    return emitOpError() << "Invalid stride " << stride << " for dimension "
                         << dim << " with size " << dimSize
                         << ". stride must be less than or equal to the "
                            "dimension size";
  }

  int32_t begin = getBegin();
  int32_t length = getLength();
  if (begin < 0 || begin >= dimSize) {
    return emitOpError() << "Invalid begin index " << begin << " for dimension "
                         << dim << " with size " << dimSize
                         << ". begin must be "
                            "in the range [0, dimSize)";
  }

  if (length < 1 || length > stride) {
    return emitOpError() << "Invalid length " << length << " for begin index "
                         << begin << " and stride " << stride
                         << " for dimension " << dim << " with size " << dimSize
                         << ". stride must be greater than or equal to length";
  }

  if (begin + length > dimSize) {
    return emitOpError() << "Invalid length " << length << " for begin index "
                         << begin << " and dimension " << dim << " with size "
                         << dimSize
                         << ". begin + length must be less than or "
                            "equal to the dimension size";
  }

  // Get the number of slices as the number of times the stride fits in the
  // dimension size starting from the begin index.
  int32_t numSlices = (dimSize - begin + stride - 1) / stride;
  int32_t totalLength = 0;
  for (int32_t i = 0; i < numSlices; i++) {
    int32_t newBegin = begin + i * stride;
    int32_t newEnd = std::min(newBegin + length, dimSize);
    totalLength += newEnd - newBegin;
  }

  if (totalLength != outputType.getDimSize(dim)) {
    return emitOpError() << "Sum of all slices must be equal to the output "
                            "dimension size for the given dimension. Expected "
                            "output dimension size: "
                         << outputType.getDimSize(dim) << ", but got "
                         << totalLength;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

// SqueezeOp verification
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

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// TransposeOp verification
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

mlir::OpFoldResult foldIdentityTranspose(mlir::tt::ttir::TransposeOp op) {
  // TransposeOp can be removed if the both 'dim0' and 'dim1' are the same.
  if (op.getDim0() == op.getDim1()) {
    return op.getInput();
  }
  return nullptr;
}

mlir::OpFoldResult sortTansposeDims(mlir::tt::ttir::TransposeOp op) {
  // Rewrite a transpose of to a canonical form where the 'dim0' is less than
  // 'dim1'.
  if (op.getDim0() > op.getDim1()) {
    auto tmp = op.getDim0();
    op.setDim0(op.getDim1());
    op.setDim1(tmp);
    return op.getResult();
  }
  return nullptr;
}

mlir::OpFoldResult forcePositiveTransposeDims(mlir::tt::ttir::TransposeOp op) {
  // Rewrite tranpose dims to a canonical form where the 'dim0' and 'dim1' are
  // in range [0, N), where N is a rank of input tensor.
  if (op.getDim0() < 0 || op.getDim1() < 0) {
    if (op.getDim0() < 0) {
      op.setDim0(op.getDim0() + op.getInput().getType().getRank());
    }
    if (op.getDim1() < 0) {
      op.setDim1(op.getDim1() + op.getInput().getType().getRank());
    }
    return op.getResult();
  }
  return nullptr;
}

mlir::OpFoldResult foldInverseTransposeOperand(mlir::tt::ttir::TransposeOp op) {
  // Transposing twice in the row over the same dimensions results in identity,
  // hence y = T(T(x)) can be replaced with y = x.
  if (auto producerOp =
          op.getInput().getDefiningOp<mlir::tt::ttir::TransposeOp>()) {
    if (op.getDim0() == producerOp.getDim0() &&
        op.getDim1() == producerOp.getDim1()) {
      return producerOp.getInput();
    }
  }
  return nullptr;
}

// TransposeOp folder
mlir::OpFoldResult mlir::tt::ttir::TransposeOp::fold(FoldAdaptor adaptor) {
  if (auto foldResult = foldIdentityTranspose(*this)) {
    return foldResult;
  }

  if (auto foldResult = sortTansposeDims(*this)) {
    return foldResult;
  }

  if (auto foldResult = forcePositiveTransposeDims(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldInverseTransposeOperand(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//

// TypecastOp folder
::llvm::LogicalResult mlir::tt::ttir::TypecastOp::fold(
    FoldAdaptor adaptor,
    ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {

  if (getType(0) == getInputs()[0].getType()) {
    results.push_back(getInputs()[0]);
    return llvm::success();
  }
  return llvm::failure();
}

//===----------------------------------------------------------------------===//
// UnsqueezeOp
//===----------------------------------------------------------------------===//

// UnsqueezeOp verification
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

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

// EmbeddingOp verification
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

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

// EmbeddingBackwardOp verification
::mlir::LogicalResult mlir::tt::ttir::EmbeddingBackwardOp::verify() {
  ::mlir::RankedTensorType weightType = getWeight().getType();
  ::mlir::RankedTensorType inputGradType = getInGradient().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

  // weightType must have rank of 2: (dictionary_size, embedding_size).
  if (weightType.getRank() != 2) {
    return emitOpError("Input must be a 2D tensor");
  }

  // inputGradType checks.
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
// ToLayoutOp
//===----------------------------------------------------------------------===//

static ::mlir::LogicalResult
verifyLayoutOp(mlir::Operation *op, mlir::Type inputTensorOrMemrefTy,
               mlir::Type outputTensorOrMemrefTy, bool allowFormatChange,
               bool allowMemorySpaceChange, bool checkMemrefRank = false,
               bool checkMemrefGridShardForm = false,
               bool checkMemrefGridShape = false,
               bool checkMemrefShardShape = false) {
  if (mlir::RankedTensorType inputTy =
          mlir::dyn_cast<mlir::RankedTensorType>(inputTensorOrMemrefTy)) {
    mlir::RankedTensorType outputTy =
        mlir::dyn_cast<mlir::RankedTensorType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    if (!inputTy.getEncoding() ||
        !mlir::isa<mlir::tt::MetalLayoutAttr>(inputTy.getEncoding())) {
      // If the input tensor does not have a layout, we can early exit.
      return mlir::success();
    }

    auto inputLayout =
        mlir::cast<mlir::tt::MetalLayoutAttr>(inputTy.getEncoding());
    auto outputLayout =
        mlir::cast<mlir::tt::MetalLayoutAttr>(outputTy.getEncoding());
    bool isFormatChange =
        inputLayout.getElementType() != outputLayout.getElementType();
    if (isFormatChange && !allowFormatChange) {
      return op->emitOpError(
          "Input and output layout element types must be the same");
    }

    bool isMemorySpaceChange =
        inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output layout memory spaces must be the same");
    }
    return mlir::success();
  }

  if (mlir::MemRefType inputTy =
          mlir::dyn_cast<mlir::MemRefType>(inputTensorOrMemrefTy)) {
    mlir::MemRefType outputTy =
        mlir::dyn_cast<mlir::MemRefType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    bool isFormatChange = inputTy.getElementType() != outputTy.getElementType();
    if (!allowFormatChange && isFormatChange) {
      return op->emitOpError(
          "Input and output layout element types must be the same");
    }

    bool isMemorySpaceChange =
        inputTy.getMemorySpace() != outputTy.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output memref memory spaces must be the same");
    }

    bool sameRank = inputTy.getRank() == outputTy.getRank();
    if (checkMemrefRank && !sameRank) {
      return op->emitOpError("Input and output memref ranks must be the same");
    }

    bool inputGridShardForm = inputTy.getShape().size() % 2 == 0;
    if (checkMemrefGridShardForm && !inputGridShardForm) {
      return op->emitOpError(
          "Input memref must be in grid-shard form, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    bool outputGridShardForm = outputTy.getShape().size() % 2 == 0;
    if (checkMemrefGridShardForm && !outputGridShardForm) {
      return op->emitOpError(
          "Output memref must be in grid-shard form, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    llvm::ArrayRef<int64_t> inputGridShape =
        inputTy.getShape().take_front(inputTy.getShape().size() / 2);
    llvm::ArrayRef<int64_t> outputGridShape =
        outputTy.getShape().take_front(outputTy.getShape().size() / 2);
    if (checkMemrefGridShape && inputGridShape != outputGridShape) {
      return op->emitOpError(
          "Input and output memref grid shapes must be the same");
    }

    return mlir::success();
  }

  return op->emitOpError("Unsupported input type for view");
}

// ToLayoutOp verification
::mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::verify() {
  return verifyLayoutOp(*this, getInput().getType(), getOutput().getType(),
                        /*allowFormatChange*/ true,
                        /*allowMemorySpaceChange*/ true);
}

// ToLayoutOp utility methods
mlir::tt::ttir::ToLayoutOp::CompoundComponents
mlir::tt::ttir::ToLayoutOp::compoundComponents() {
  CompoundComponents components;
  if (mlir::isa<mlir::RankedTensorType>(getInput().getType())) {
    auto inputLayout = mlir::cast<tt::MetalLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding());
    auto outputLayout = mlir::cast<tt::MetalLayoutAttr>(
        mlir::cast<mlir::RankedTensorType>(getOutput().getType())
            .getEncoding());
    components.isLayoutChange =
        inputLayout.getLinear() != outputLayout.getLinear();
    components.isGridChange = inputLayout.getGrid() != outputLayout.getGrid();
    bool isShardChange =
        inputLayout.getShardShape() != outputLayout.getShardShape();
    assert(components.isGridChange == isShardChange);
    components.isFormatChange =
        inputLayout.getElementType() != outputLayout.getElementType();
    components.isMemorySpaceChange =
        inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
  } else {
    auto inputMemref = mlir::cast<mlir::MemRefType>(getInput().getType());
    auto outputMemref = mlir::cast<mlir::MemRefType>(getOutput().getType());
    components.isLayoutChange = false;
    bool isShapeChange = inputMemref.getShape() != outputMemref.getShape();
    components.isGridChange = isShapeChange;
    components.isFormatChange =
        inputMemref.getElementType() != outputMemref.getElementType();
    components.isMemorySpaceChange =
        inputMemref.getMemorySpace() != outputMemref.getMemorySpace();
  }
  return components;
}

mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::fold(
    FoldAdaptor, llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  if (auto emptyOp = getInput().getDefiningOp<tensor::EmptyOp>()) {
    results.push_back(getOutput());
    return mlir::success();
  }
  return mlir::failure();
}

mlir::LogicalResult mlir::tt::ttir::ToLayoutOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  if (getNumResults() == 0) {
    return failure();
  }

  assert(getNumResults() == 1 && "ToLayoutOp should have exactly one result");

  if (!mlir::isa<::mlir::RankedTensorType>(getResult(0).getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeOutput =
      mlir::bufferization::getBuffer(rewriter, getOutput(), options);
  if (failed(maybeOutput)) {
    return maybeOutput;
  }

  rewriter.create<::mlir::tt::ttir::ToLayoutOp>(getLoc(), TypeRange(),
                                                *maybeInput, *maybeOutput);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     *maybeOutput);
  return success();
}

//===----------------------------------------------------------------------===//
// StreamLayoutOp
//===----------------------------------------------------------------------===//

void mlir::tt::ttir::StreamLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {
  patterns.add(+[](StreamLayoutOp op, mlir::PatternRewriter &rewriter) {
    ViewLayoutOp viewOp = op.getInput().getDefiningOp<ViewLayoutOp>();
    if (!viewOp) {
      return failure();
    }

    auto viewMemref = mlir::dyn_cast<MemRefType>(viewOp.getResult().getType());
    if (!viewMemref) {
      return failure();
    }

    auto currentResultMemref = mlir::cast<MemRefType>(op.getResult().getType());
    auto streamAttr = rewriter.getAttr<StreamLayoutAttr>(
        viewMemref.getLayout().getAffineMap().compose(
            currentResultMemref.getLayout().getAffineMap()));
    auto newMemref = MemRefType::get(
        currentResultMemref.getShape(), currentResultMemref.getElementType(),
        streamAttr, currentResultMemref.getMemorySpace());
    rewriter.replaceOpWithNewOp<StreamLayoutOp>(
        op, newMemref, viewOp.getInput(), op.getStorage());
    return success();
  });
}

void mlir::tt::ttir::StreamLayoutOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "stream");
}

// StreamLayoutOp verification
mlir::LogicalResult mlir::tt::ttir::StreamLayoutOp::verify() {
  auto inputStorageVerification =
      verifyLayoutOp(*this, getInput().getType(), getStorage().getType(),
                     /*allowFormatChange*/ false,
                     /*allowMemorySpaceChange*/ true,
                     /*checkMemrefRank*/ true,
                     /*checkMemrefGridShardForm */ true,
                     /*checkMemrefGridShape*/ false,
                     /*checkMemrefShardShape*/ false);
  if (failed(inputStorageVerification)) {
    return inputStorageVerification;
  }

  auto storageResultVerification =
      verifyLayoutOp(*this, getStorage().getType(), getResult().getType(),
                     /*allowFormatChange*/ false,
                     /*allowMemorySpaceChange*/ true,
                     /*checkMemrefRank*/ true,
                     /*checkMemrefGridShardForm */ true,
                     /*checkMemrefGridShape*/ false,
                     /*checkMemrefShardShape*/ true);
  if (failed(storageResultVerification)) {
    return storageResultVerification;
  }

  MemRefType inputMemrefType = mlir::dyn_cast<MemRefType>(getInput().getType());
  MemRefType resultMemrefType =
      mlir::dyn_cast<MemRefType>(getResult().getType());
  if (inputMemrefType && resultMemrefType &&
      (inputMemrefType.getMemorySpace() != resultMemrefType.getMemorySpace())) {
    return this->emitOpError(
        "Input and result memref memory spaces must be the same");
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttir::StreamLayoutOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  if (!mlir::isa<::mlir::RankedTensorType>(getResult().getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeStorage =
      mlir::bufferization::getBuffer(rewriter, getStorage(), options);
  if (failed(maybeStorage)) {
    return maybeStorage;
  }

  ::llvm::SmallVector<mlir::Value> invocationStack;
  Value result = rewriter.create<::mlir::tt::ttir::StreamLayoutOp>(
      getLoc(), *getBufferType(getResult(), options, invocationStack),
      *maybeInput, *maybeStorage);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this, result);
  return success();
}

//===----------------------------------------------------------------------===//
// ViewLayoutOp
//===----------------------------------------------------------------------===//

// ViewLayoutOp verification
mlir::LogicalResult mlir::tt::ttir::ViewLayoutOp::verify() {
  return verifyLayoutOp(*this, getInput().getType(), getResult().getType(),
                        /*allowFormatChange*/ false,
                        /*allowMemorySpaceChange*/ false);
}

void mlir::tt::ttir::ViewLayoutOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "view");
}

bool mlir::tt::ttir::ViewLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an input, it is a bufferized to a memory read.
  return false;
}

bool mlir::tt::ttir::ViewLayoutOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an output, it is a bufferized to a memory write.
  return false;
}

mlir::LogicalResult mlir::tt::ttir::ViewLayoutOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  if (mlir::isa<mlir::MemRefType>(getInput().getType())) {
    return mlir::failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  ::llvm::SmallVector<mlir::Value> invocationStack;
  mlir::bufferization::replaceOpWithNewBufferizedOp<
      mlir::tt::ttir::ViewLayoutOp>(
      rewriter, *this, *getBufferType(getResult(), options, invocationStack),
      *maybeInput);
  return mlir::success();
}

mlir::bufferization::AliasingValueList
mlir::tt::ttir::ViewLayoutOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType>
mlir::tt::ttir::ViewLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    ::llvm::SmallVector<mlir::Value> &) {
  auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
  return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
      .getBufferType();
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

// LinearOp verification
::mlir::LogicalResult mlir::tt::ttir::LinearOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  std::optional<::mlir::RankedTensorType> biasType =
      getBias() ? std::make_optional(getBias().getType()) : std::nullopt;
  ::mlir::RankedTensorType outputType = getOutput().getType();

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
    if (!mlir::OpTrait::util::getBroadcastedShape(
            inputABatchDims, inputBBatchDims, broadcastedShape)) {

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
    if (!mlir::OpTrait::util::getBroadcastedShape(matmulShape, biasShape,
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

// If value is defined by TransposeOp with transpose dimensions
// (rank - 2, rank - 1), return the input of the TransposeOp, otherwise return
// std::nullopt. This is used for canonicalization of MatmulOp and LinearOp.
static std::optional<mlir::TypedValue<mlir::RankedTensorType>>
getTransposeOpOperand(mlir::TypedValue<mlir::RankedTensorType> value) {
  auto producerOp = value.getDefiningOp<mlir::tt::ttir::TransposeOp>();
  if (!producerOp) {
    return std::nullopt;
  }

  int64_t rank = value.getType().getRank();
  if (rank < 2 || producerOp.getDim0() != rank - 2 ||
      producerOp.getDim1() != rank - 1) {
    return std::nullopt;
  }

  return producerOp.getInput();
}

// LinearOp canonicalization
void mlir::tt::ttir::LinearOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // If bias is not provided, linear operation is equivalent to matmul.
  patterns.add(+[](ttir::LinearOp op, mlir::PatternRewriter &rewriter) {
    if (!op.getBias()) {
      rewriter.replaceOpWithNewOp<ttir::MatmulOp>(
          op, op.getType(), op.getA(), op.getB(), op.getOutput(),
          op.getTransposeA(), op.getTransposeB());
      return mlir::success();
    }
    return mlir::failure();
  });

  // linear(transpose(a), b, bias transpose_a, transpose_b) ->
  //   linear(a, b, bias, !transpose_a, transpose_b)
  patterns.add(+[](ttir::LinearOp op, mlir::PatternRewriter &rewriter) {
    auto inputACanonical = getTransposeOpOperand(op.getA());
    if (!inputACanonical) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ttir::LinearOp>(
        op, op.getType(), *inputACanonical, op.getB(), op.getBias(),
        op.getOutput(), !op.getTransposeA(), op.getTransposeB());

    return mlir::success();
  });

  // linear(a, transpose(b), bias transpose_a, transpose_b) ->
  //   linear(a, b, bias, transpose_a, !transpose_b)
  patterns.add(+[](ttir::LinearOp op, mlir::PatternRewriter &rewriter) {
    auto inputBCanonical = getTransposeOpOperand(op.getB());
    if (!inputBCanonical) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ttir::LinearOp>(
        op, op.getType(), op.getA(), *inputBCanonical, op.getBias(),
        op.getOutput(), op.getTransposeA(), !op.getTransposeB());

    return mlir::success();
  });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// ANCHOR: adding_an_op_matmul_ttir_verify
// MatmulOp verification
::mlir::LogicalResult mlir::tt::ttir::MatmulOp::verify() {
  ::mlir::RankedTensorType inputAType = getA().getType();
  ::mlir::RankedTensorType inputBType = getB().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

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
    if (!mlir::OpTrait::util::getBroadcastedShape(
            inputABatchDims, inputBBatchDims, broadcastedShape)) {

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
// ANCHOR_END: adding_an_op_matmul_ttir_verify

// MatmulOp canonicalization
void mlir::tt::ttir::MatmulOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // matmul(transpose(a), b, transpose_a, transpose_b) ->
  //   matmul(a, b, !transpose_a, transpose_b)
  patterns.add(+[](ttir::MatmulOp op, mlir::PatternRewriter &rewriter) {
    auto inputACanonical = getTransposeOpOperand(op.getA());
    if (!inputACanonical) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ttir::MatmulOp>(
        op, op.getType(), *inputACanonical, op.getB(), op.getOutput(),
        !op.getTransposeA(), op.getTransposeB());

    return mlir::success();
  });

  // matmul(a, transpose(b), transpose_a, transpose_b) ->
  //   matmul(a, b, transpose_a, !transpose_b)
  patterns.add(+[](ttir::MatmulOp op, mlir::PatternRewriter &rewriter) {
    auto inputBCanonical = getTransposeOpOperand(op.getB());
    if (!inputBCanonical) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ttir::MatmulOp>(
        op, op.getType(), op.getA(), *inputBCanonical, op.getOutput(),
        op.getTransposeA(), !op.getTransposeB());

    return mlir::success();
  });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// UpsampleOp
//===----------------------------------------------------------------------===//

// UpsampleOp verification
::mlir::LogicalResult mlir::tt::ttir::Upsample2dOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();

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

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
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
// AllocOp
//===----------------------------------------------------------------------===//

// AllocOp verification
::mlir::LogicalResult mlir::tt::ttir::AllocOp::verify() {
  auto layout = mlir::dyn_cast_if_present<mlir::tt::MetalLayoutAttr>(
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

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

// RepeatOp verification.
::mlir::LogicalResult mlir::tt::ttir::RepeatOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
  llvm::ArrayRef<int64_t> repeatDimensions = getRepeatDimensions();

  // Input tensor and repeat dimension argument must have same rank.
  if (inputType.getRank() != static_cast<int64_t>(repeatDimensions.size())) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the number of repeat dimensions "
                         << repeatDimensions.size() << ".";
  }

  // Input and output tensors must have the same rank.
  if (inputType.getRank() != outputType.getRank()) {
    return emitOpError() << "Input tensor rank " << inputType.getRank()
                         << " doesn't match the output tensor rank "
                         << outputType.getRank() << ".";
  }

  // Verify output shape based on input shape and repeat dimension argument.
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();

  for (size_t i = 0; i < inputShape.size(); i++) {
    // Verify that the repeat dimension is greater than 0.
    if (repeatDimensions[i] <= 0) {
      return emitOpError() << "Repeat dimension at index " << i
                           << " must be greater than 0.";
    }

    int64_t expectedDimValue = inputShape[i] * repeatDimensions[i];
    if (expectedDimValue != outputShape[i]) {
      return emitOpError() << "Input tensor shape ("
                           << ttmlir::utils::join(inputShape, ",")
                           << ") at index " << i
                           << " does not repeat to output ("
                           << ttmlir::utils::join(outputShape, ",")
                           << ") using repeat value " << repeatDimensions[i]
                           << ".";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

// RepeatInterleaveOp verification
::mlir::LogicalResult mlir::tt::ttir::RepeatInterleaveOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::RankedTensorType outputType = getOutput().getType();
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

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

// AllGatherOp verification
::mlir::LogicalResult mlir::tt::ttir::AllGatherOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  int32_t gatherDim = getAllGatherDim();

  if (gatherDim >= inputType.getRank() || gatherDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid dimension for all gather op. Gather dimension must be "
               ">= to "
               "input tensor rank or < -input tensor rank, got gather_dim = ")
           << gatherDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

// AllReduceOp verification
::mlir::LogicalResult mlir::tt::ttir::AllReduceOp::verify() {
  ::mlir::tt::ReduceType reduceType = getReduceType();

  // Currently TTIR only supports the following reduce types.
  if (reduceType != ::mlir::tt::ReduceType::Sum &&
      reduceType != ::mlir::tt::ReduceType::Max &&
      reduceType != ::mlir::tt::ReduceType::Min) {
    return emitOpError("Invalid reduction op for all reduce op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

// ReduceScatterOp verification
::mlir::LogicalResult mlir::tt::ttir::ReduceScatterOp::verify() {
  ::mlir::RankedTensorType inputType = getInput().getType();
  ::mlir::tt::ReduceType reduceType = getReduceType();
  int32_t scatterDim = getScatterDim();

  // Currently TTIR only supports the following reduce types.
  if (reduceType != ::mlir::tt::ReduceType::Sum &&
      reduceType != ::mlir::tt::ReduceType::Max &&
      reduceType != ::mlir::tt::ReduceType::Min) {
    return emitOpError("Invalid reduction op for reduce scatter op.");
  }

  if (scatterDim >= inputType.getRank() || scatterDim < -inputType.getRank()) {
    return emitOpError(
               "Invalid dimension for reduce scatter op. Scatter dimension "
               "must be >= to input tensor rank or < -input tensor rank, got "
               "scatter_dim = ")
           << scatterDim;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

// CollectivePermuteOp verification
::mlir::LogicalResult mlir::tt::ttir::CollectivePermuteOp::verify() {
  auto sourceTargetPairs = getSourceTargetPairs().getValues<int64_t>();

  // Check that the rank of sourceTargetPairs is 2D.
  llvm::ArrayRef<int64_t> sourceTargetPairsShape =
      getSourceTargetPairs().getType().getShape();
  const size_t sourceTargetPairsRank = sourceTargetPairsShape.size();

  if (sourceTargetPairsRank != 2) {
    return emitOpError("The rank of source target pairs must be 2, got rank = ")
           << sourceTargetPairsRank;
  }

  /* Check that the 'src' values and 'dest' values in sourceTargetPairs is
  unique. Given a 2D rank tensor of source target pairs eg. [['src', 'target'],
  ['src', 'target'] ...], we need to ensure that each 'src' is unique and each
  'target' is unique.
  */
  auto areElementsUnique = [](const auto &sourceTargetPairs) -> bool {
    for (size_t i = 0; i < sourceTargetPairs.size(); i++) {
      int target = sourceTargetPairs[i];
      for (size_t j = i + 2; j < sourceTargetPairs.size(); j += 2) {
        if (sourceTargetPairs[j] == target) {
          return false;
        }
      }
    }

    return true;
  };

  if (!areElementsUnique(sourceTargetPairs)) {
    return emitOpError(
        "There are duplicate 'src' or 'dest' devices in source target pairs");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MeshShardOp
//===----------------------------------------------------------------------===//

// MeshShardOp verification
::mlir::LogicalResult mlir::tt::ttir::MeshShardOp::verify() {
  auto shardType = getShardType();

  // Currently, we are not supporting maximal from StableHLO.
  if (shardType == mlir::tt::MeshShardType::Maximal) {
    return emitOpError("Invalid shard_type (maximal) for mesh_shard op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::ScatterOp::verify() {

  ArrayRef<int64_t> inputShape =
      mlir::cast<RankedTensorType>(getInput().getType()).getShape();

  if (getUpdateWindowDims().size() + getInsertedWindowDims().size() !=
      inputShape.size()) {
    return emitOpError("Batching currently not supported");
  }

  for (uint64_t insertedWindowDims : getInsertedWindowDims()) {
    if (inputShape[insertedWindowDims] != 1) {
      return emitOpError("Dimension size to slice into must be 1");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::UpdateCacheOp::verify() {
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

::mlir::LogicalResult mlir::tt::ttir::FillCacheOp::verify() {
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
// ReverseOp
//===----------------------------------------------------------------------===//

// ReverseOp verification
::mlir::LogicalResult mlir::tt::ttir::ReverseOp::verify() {
  llvm::ArrayRef<int64_t> dimensions = getDimensions();

  // Check that all given dimensions are unique/not repeating.
  llvm::SmallDenseSet<int64_t> uniqueDims(dimensions.begin(), dimensions.end());

  if (uniqueDims.size() != dimensions.size()) {
    return emitOpError("dimensions should be unique. Got: ") << dimensions;
  }

  ::mlir::RankedTensorType operandTy = getInput().getType();

  // Check that each dimension is positive and within valid interval [0,
  // operandRank).
  for (int64_t dim : dimensions) {
    if (dim < 0) {
      return emitOpError(
                 "all dimensions should be non-negative. Got dimension: ")
             << dim;
    }

    if (dim >= operandTy.getRank()) {
      return emitOpError("all dimensions should be in interval [0, ")
             << operandTy.getRank() << "). Got dimension: " << dim;
    }
  }

  return success();
}

// ReverseOp canonicalization
void mlir::tt::ttir::ReverseOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // Reverse dimensions of two consecutive ReverseOps can be folded into a
  // single ReverseOp where the dimensions are the symmetric difference of the
  // two sets of dimensions.
  patterns.add(+[](mlir::tt::ttir::ReverseOp op,
                   mlir::PatternRewriter &rewriter) {
    auto producerOp = op.getInput().getDefiningOp<ttir::ReverseOp>();
    if (!producerOp) {
      return mlir::failure();
    }

    llvm::SmallBitVector reverseDimensions(op.getInput().getType().getRank());
    llvm::for_each(op.getDimensions(), [&reverseDimensions](int64_t dim) {
      reverseDimensions.flip(dim);
    });
    llvm::for_each(
        producerOp.getDimensions(),
        [&reverseDimensions](int64_t dim) { reverseDimensions.flip(dim); });

    llvm::SmallVector<int64_t> setIndices;
    llvm::copy_if(llvm::seq<int64_t>(reverseDimensions.size()),
                  std::back_inserter(setIndices),
                  [&](int64_t i) { return reverseDimensions.test(i); });

    rewriter.replaceOpWithNewOp<ttir::ReverseOp>(
        op, op.getType(), producerOp.getInput(), op.getOutput(), setIndices);
    return success();
  });

  // ReverseOp with empty reverse dimensions is a no-op.
  patterns.add(
      +[](mlir::tt::ttir::ReverseOp op, mlir::PatternRewriter &rewriter) {
        if (!op.getDimensions().empty()) {
          return mlir::failure();
        }

        rewriter.replaceAllOpUsesWith(op, op.getInput());
        return mlir::success();
      });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

//===----------------------------------------------------------------------===//
// PermuteOp
//===----------------------------------------------------------------------===//

// PermuteOp verification
::mlir::LogicalResult mlir::tt::ttir::PermuteOp::verify() {
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
    return emitOpError("Expected result shape (")
           << ttmlir::utils::join(expectedResultShape, ", ") << "), got ("
           << ttmlir::utils::join(resultShape, ", ") << ")";
  }

  return success();
}

mlir::OpFoldResult foldIdentityPermute(mlir::tt::ttir::PermuteOp op) {
  // PermuteOp with identity permutation is a no-op.
  // The input can be used directly as the output.
  if (llvm::is_sorted(op.getPermutation())) {
    return op.getInput();
  }
  return nullptr;
}

mlir::OpFoldResult foldOperandPermute(mlir::tt::ttir::PermuteOp op) {
  // If the producer is a PermuteOp we can compose the permutation attributes
  // into `op`, and set the input to the producers input.
  if (auto producerOp =
          op.getInput().getDefiningOp<mlir::tt::ttir::PermuteOp>()) {
    llvm::SmallVector<int64_t> composedPermutation =
        ttmlir::utils::applyPermutation(producerOp.getPermutation(),
                                        op.getPermutation());
    op.setPermutation(composedPermutation);
    op->setOperand(0, producerOp.getInput());
    return op.getResult();
  }
  return nullptr;
}

// PermuteOp folder
mlir::OpFoldResult mlir::tt::ttir::PermuteOp::fold(FoldAdaptor adaptor) {

  if (auto foldResult = foldIdentityPermute(*this)) {
    return foldResult;
  }

  if (auto foldResult = foldOperandPermute(*this)) {
    return foldResult;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

// GenericOp verification
::mlir::LogicalResult mlir::tt::ttir::GenericOp::verify() {
  if (!getGrid().getMapping().isEmpty()) {
    return emitOpError("GenericOp grid mapping is not supported");
  }

  if (getOutputs().size() != 1) {
    return emitOpError(
        "GenericOp must currently have exactly one output operand");
  }

  // Output grid shape must equal the GenericOp grid shape.
  auto opGridShape = getGrid().getShape();
  for (auto output : getOutputs()) {
    Type operandType = output.getType();
    ArrayRef<int64_t> outputGridShape;
    if (RankedTensorType tensorType =
            mlir::dyn_cast<RankedTensorType>(operandType)) {
      if (!tensorType.getEncoding()) {
        // Skip layout checks if the tensor type does not have a layout yet.
        continue;
      }
      MetalLayoutAttr layout =
          mlir::cast<MetalLayoutAttr>(tensorType.getEncoding());
      outputGridShape = layout.getGrid().getShape();
    } else {
      auto memref = mlir::cast<MemRefType>(operandType);
      // If the top level operand is a memref, the front half of its shape
      // is the grid shape, so we cut it off the back to get just the grid
      // shape.
      assert(memref.getRank() % 2 == 0);
      outputGridShape = memref.getShape().take_front(memref.getRank() / 2);
    }
    if (opGridShape != outputGridShape) {
      return emitOpError(
          "Output grid shape must match the GenericOp grid shape");
    }
  }

  ValueTypeRange<OperandRange> operandTypes = getOperation()->getOperandTypes();
  for (Region &region : getRegions()) {
    if (!region.hasOneBlock()) {
      return emitOpError("GenericOp region must have a single block");
    }

    if (region.getNumArguments() < this->getNumOperands()) {
      return emitOpError("GenericOp region must have at least as many "
                         "arguments as the number of top-level operands");
    }

    auto memrefArguments =
        region.getArguments().take_front(operandTypes.size());
    for (BlockArgument arg : memrefArguments) {
      auto blockMemref = mlir::dyn_cast<MemRefType>(arg.getType());
      if (!blockMemref) {
        return emitOpError("GenericOp region arguments must be of MemRefType");
      }

      Type operandType = operandTypes[arg.getArgNumber()];
      Attribute expectedMemorySpace;
      ArrayRef<int64_t> expectedShardShape;
      bool isStream = false;
      if (RankedTensorType tensorType =
              mlir::dyn_cast<RankedTensorType>(operandType)) {
        if (!tensorType.getEncoding()) {
          // Skip layout checks if the tensor type does not have a layout yet
          continue;
        }
        MetalLayoutAttr layout =
            mlir::cast<MetalLayoutAttr>(tensorType.getEncoding());
        expectedMemorySpace = layout.getMemref().getMemorySpace();
        expectedShardShape = layout.getMemref().getShape();
      } else {
        auto memref = mlir::cast<MemRefType>(operandType);
        expectedMemorySpace = memref.getMemorySpace();
        // If the top level operand is a memref, the front half of its shape
        // will include the grid shape, so we cut it off to get just the shard
        // shape.
        assert(memref.getRank() % 2 == 0);
        expectedShardShape = memref.getShape().take_back(memref.getRank() / 2);
        isStream = mlir::isa<tt::StreamLayoutAttr>(memref.getLayout());
      }

      if (!isStream && expectedMemorySpace != blockMemref.getMemorySpace()) {
        return emitOpError("GenericOp region argument memory space must match "
                           "the memory space of the corresponding operand");
      }

      if (expectedShardShape != blockMemref.getShape()) {
        return emitOpError("GenericOp region argument shape must match the "
                           "shape of the corresponding operand");
      }
    }

    auto additionalArguments =
        region.getArguments().drop_front(operandTypes.size());
    for (BlockArgument arg : additionalArguments) {
      bool supportedType = mlir::isa<SemaphoreType>(arg.getType());
      if (!supportedType) {
        return emitOpError(
            "Additional GenericOp region arguments must be of SemaphoreType");
      }
    }
  }

  return success();
}

void mlir::tt::ttir::GenericOp::getAsmBlockArgumentNames(
    Region &region, function_ref<void(Value, StringRef)> setNameFn) {
  int cbIndex = 0;
  int semIndex = 0;
  for (BlockArgument arg : region.getArguments()) {
    if (mlir::isa<MemRefType>(arg.getType())) {
      setNameFn(arg, "cb" + std::to_string(cbIndex++));
    } else if (mlir::isa<SemaphoreType>(arg.getType())) {
      setNameFn(arg, "sem" + std::to_string(semIndex++));
    } else {
      llvm_unreachable("Unexpected region argument type");
    }
  }
}

void mlir::tt::ttir::GenericOp::getAsmBlockNames(
    function_ref<void(Block *, StringRef)> setNameFn) {
  size_t numRegions = getNumRegions();
  for (Region &region : getRegions()) {
    // Right now the last region is implicitly the compute region.
    if (region.getRegionNumber() < (numRegions - 1)) {
      setNameFn(&region.front(),
                "datamovement" + std::to_string(region.getRegionNumber()));
    } else {
      setNameFn(&region.front(), "compute");
    }
  }
}

mlir::LogicalResult mlir::tt::ttir::GenericOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  if (getNumResults() == 0) {
    return failure();
  }

  assert(getNumResults() == 1 && "GenericOp should have exactly one result");
  assert(getOutputs().size() == 1 &&
         "GenericOp should have exactly one output");

  if (!mlir::isa<mlir::RankedTensorType>(getResult(0).getType())) {
    return failure();
  }
  mlir::SmallVector<mlir::Value> bufferInputs;
  bufferInputs.reserve(getInputs().size());
  for (auto input : getInputs()) {
    auto maybeValue = bufferization::getBuffer(rewriter, input, options);
    if (failed(maybeValue)) {
      return maybeValue;
    }
    bufferInputs.push_back(*maybeValue);
  }
  mlir::SmallVector<mlir::Value> bufferOutputs;
  bufferOutputs.reserve(getOutputs().size());
  for (auto output : getOutputs()) {
    auto maybeValue = bufferization::getBuffer(rewriter, output, options);
    if (failed(maybeValue)) {
      return maybeValue;
    }
    bufferOutputs.push_back(*maybeValue);
  }
  auto bufferGeneric = rewriter.create<mlir::tt::ttir::GenericOp>(
      getLoc(), ValueRange(), bufferInputs, bufferOutputs, getGrid(),
      getIndexingMaps(), getIteratorTypes(), getNumRegions());
  for (mlir::Region &region : bufferGeneric.getRegions()) {
    region.takeBody(getRegion(region.getRegionNumber()));
  }
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     bufferOutputs);
  return success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

// Common verifier for all Reduce ops.
static mlir::LogicalResult
verifyReduceOp(llvm::function_ref<mlir::InFlightDiagnostic()> emitOpError,
               mlir::RankedTensorType inputType,
               const std::optional<mlir::ArrayAttr> &reduceDims, bool keepDim,
               ::llvm::ArrayRef<int64_t> specifiedOutputShape) {

  int64_t inputTensorRank = inputType.getRank();

  llvm::BitVector reduceDimsMask(inputTensorRank, false);
  if (!reduceDims) {
    reduceDimsMask.set();
  } else {
    llvm::SmallSet<int64_t, 4> uniqueReduceDims;
    for (mlir::Attribute reduceDim : *reduceDims) {
      int64_t reduceDimInt = mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
      if (reduceDimInt < -inputTensorRank || reduceDimInt >= inputTensorRank) {
        return emitOpError() << "Reduce dimension " << reduceDimInt
                             << " is out of range for input tensor of rank "
                             << inputTensorRank;
      }
      uniqueReduceDims.insert(reduceDimInt);
      reduceDimsMask.set((reduceDimInt + inputTensorRank) % inputTensorRank);
    }

    if (uniqueReduceDims.size() != reduceDims->size()) {
      return emitOpError() << "Reduce dimensions are not unique";
    }
  }

  // Check that the output shape is valid.
  llvm::SmallVector<int64_t> expectedOutputShape;
  for (int64_t index = 0; index < inputTensorRank; ++index) {
    if (!reduceDimsMask[index]) {
      expectedOutputShape.push_back(inputType.getDimSize(index));
    } else if (keepDim) {
      expectedOutputShape.push_back(1);
    }
  }

  // Cover edge case where all dims are reduced, and keepDim==false.
  if (expectedOutputShape.empty() && !keepDim) {
    expectedOutputShape.push_back(1);
  }

  // Finally, compare shapes.
  if (!llvm::equal(specifiedOutputShape, expectedOutputShape)) {
    return emitOpError() << "Expected output shape ("
                         << ttmlir::utils::join(expectedOutputShape, ", ")
                         << "), got ("
                         << ttmlir::utils::join(specifiedOutputShape, ", ")
                         << ")";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

// MaxOp verification.
::mlir::LogicalResult mlir::tt::ttir::MaxOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

// MeanOp verification.
::mlir::LogicalResult mlir::tt::ttir::MeanOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

// SumOp verification.
::mlir::LogicalResult mlir::tt::ttir::SumOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce MinOp
//===----------------------------------------------------------------------===//

// MinOp verification.
::mlir::LogicalResult mlir::tt::ttir::MinOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ProdOp
//===----------------------------------------------------------------------===//

// ProdOp verification.
::mlir::LogicalResult mlir::tt::ttir::ProdOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// ReduceAndOp
//===----------------------------------------------------------------------===//

// ReduceAndOp verification.
::mlir::LogicalResult mlir::tt::ttir::ReduceAndOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// ReduceOrOp
//===----------------------------------------------------------------------===//

// ReduceOrOp verification.
::mlir::LogicalResult mlir::tt::ttir::ReduceOrOp::verify() {
  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// Reduce ArgMaxOp
//===----------------------------------------------------------------------===//

// ArgMaxOp verification.
::mlir::LogicalResult mlir::tt::ttir::ArgMaxOp::verify() {
  auto dimArg = getDimArg();
  if (dimArg && dimArg->size() > 1) {
    return getOperation()->emitOpError()
           << "can only reduce one dimension; number of specified dimensions: "
           << dimArg->size() << ".";
  }

  return verifyReduceOp([&]() { return emitOpError(); }, getInput().getType(),
                        getDimArg(), getKeepDim(), getType().getShape());
}

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::tt::ttir::CumSumOp::verify() {
  int64_t dim = getDim();
  int64_t inputRank = getInput().getType().getRank();
  if (dim < 0 || dim >= inputRank) {
    return emitOpError() << "specified dimension should be between 0 and "
                         << (inputRank - 1) << ", but got: " << dim << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// YieldOp / AwaitOp
//===----------------------------------------------------------------------===//

static bool valueInBlockArguments(mlir::Value value, mlir::Block *block) {
  return llvm::is_contained(block->getArguments(), value);
}

static mlir::Value recurseThroughMemrefCollapse(mlir::Value value) {
  while (auto memrefCastOp =
             value.getDefiningOp<::mlir::memref::CollapseShapeOp>()) {
    value = memrefCastOp.getOperand();
  }
  return value;
}

static ::mlir::LogicalResult operandsInBlockArguments(mlir::Operation *op,
                                                      mlir::Block *block) {
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    mlir::Value value = recurseThroughMemrefCollapse(operand.get());
    if (!valueInBlockArguments(value, block)) {
      return op->emitOpError() << "operand[" << operand.getOperandNumber()
                               << "] not in block arguments";
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult mlir::tt::ttir::YieldOp::verify() {
  return operandsInBlockArguments(getOperation(), getOperation()->getBlock());
}

::mlir::LogicalResult mlir::tt::ttir::AwaitOp::verify() {
  return operandsInBlockArguments(getOperation(), getOperation()->getBlock());
}
