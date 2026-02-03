// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Utils/VerificationUtils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir::verification_utils::conv2d {

std::tuple<InputTensorDims, WeightTensorDims, std::optional<BiasTensorDims>>
getConv2dInputDims(mlir::tt::ttir::Conv2dOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfoAttr();
  InputTensorDims inputDims;
  auto inputType = op->getInput().getType();
  if (flatInfo) {
    inputDims = {flatInfo.getBatchSize(), flatInfo.getInputHeight(),
                 flatInfo.getInputWidth(),
                 inputType.getDimSize(op->getChannelDim())};
  } else {

    inputDims = {inputType.getDimSize(op->getBatchDim()),
                 inputType.getDimSize(op->getHeightDim()),
                 inputType.getDimSize(op->getWidthDim()),
                 inputType.getDimSize(op->getChannelDim())};
  }

  auto weightType = op->getWeight().getType();
  WeightTensorDims weightDims = {
      weightType.getDimSize(llvm::to_underlying(WeightDim::WEIGHT_OUT_CHANNEL)),
      weightType.getDimSize(llvm::to_underlying(WeightDim::WEIGHT_IN_CHANNEL)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim::WEIGHT_KERNEL_HEIGHT)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim::WEIGHT_KERNEL_WIDTH))};

  std::optional<BiasTensorDims> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(op->getChannelDim())};
  }

  return {inputDims, weightDims, biasDims};
}

OutputTensorDims getConv2dOutputDims(mlir::tt::ttir::Conv2dOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfoAttr();
  OutputTensorDims outputDims;
  auto outputType = op->getType();
  if (flatInfo) {
    outputDims.flattenedDim = outputType.getDimSize(FLATTENED_DIM);
    outputDims.outputChannels = outputType.getDimSize(op->getChannelDim());
  } else {
    outputDims.batchSize = outputType.getDimSize(op->getBatchDim());
    outputDims.outputHeight = outputType.getDimSize(op->getHeightDim());
    outputDims.outputWidth = outputType.getDimSize(op->getWidthDim());
    outputDims.outputChannels = outputType.getDimSize(op->getChannelDim());
  }

  return outputDims;
}

llvm::Expected<Conv2dParams> getConv2dParams(mlir::tt::ttir::Conv2dOp *op) {
  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(op->getStrideAttr());
  if (!stride) {
    return llvm::createStringError(llvm::toString(stride.takeError()) +
                                   " for stride");
  }

  auto padding =
      ttmlir::utils::getQuadrupleOfInteger<int32_t>(op->getPaddingAttr());
  if (!padding) {
    return llvm::createStringError(llvm::toString(padding.takeError()) +
                                   " for padding");
  }

  auto dilation =
      ttmlir::utils::getPairOfInteger<int32_t>(op->getDilationAttr());
  if (!dilation) {
    return llvm::createStringError(llvm::toString(dilation.takeError()) +
                                   " for dilation");
  }

  return Conv2dParams{Spatial2DParam(*stride), Spatial4DParam(*padding),
                      Spatial2DParam(*dilation), op->getGroups()};
}

mlir::LogicalResult verifyConv2dParams(mlir::tt::ttir::Conv2dOp *op,
                                       const Conv2dParams &params) {
  auto isPositive = [](int64_t v) { return v > 0; };
  auto isNonNegative = [](int64_t v) { return v >= 0; };

  if (!isPositive(params.stride.vertical) ||
      !isPositive(params.stride.horizontal)) {
    return op->emitOpError("Stride attribute values must be > 0.");
  }

  if (!isPositive(params.dilation.vertical) ||
      !isPositive(params.dilation.horizontal)) {
    return op->emitOpError("Dilation attribute values must be > 0.");
  }

  if (!isNonNegative(params.padding.top) ||
      !isNonNegative(params.padding.left) ||
      !isNonNegative(params.padding.bottom) ||
      !isNonNegative(params.padding.right)) {
    return op->emitOpError("Padding attribute values must be >= 0.");
  }

  return mlir::success();
}

::mlir::LogicalResult verifyConv2dInputDims(
    mlir::tt::ttir::Conv2dOp *op, const InputTensorDims &inputDims,
    const WeightTensorDims &weightDims,
    const std::optional<BiasTensorDims> &biasDims, const Conv2dParams &params) {

  if (inputDims.inputChannels % params.groups != 0) {
    return op->emitOpError()
           << "The number of input channels from the input tensor ("
           << inputDims.inputChannels
           << ") is not divisible by the number of groups (" << params.groups
           << ").";
  }

  if (weightDims.outputChannels % params.groups != 0) {
    return op->emitOpError()
           << "The number of output channels from the weight tensor ("
           << weightDims.outputChannels
           << ") is not divisible by the number of groups (" << params.groups
           << ").";
  }

  if (inputDims.inputChannels / params.groups != weightDims.inChannPerGroup) {
    return op->emitOpError()
           << "The number of input channels per group ("
           << inputDims.inputChannels
           << ") must match the number of input channels in the weight tensor ("
           << weightDims.inChannPerGroup << ").";
  }

  if (biasDims && biasDims->outputChannels != weightDims.outputChannels) {
    return op->emitOpError()
           << "The number of output channels from the weight tensor ("
           << weightDims.outputChannels
           << ") must match the number of output channels in the bias tensor ("
           << biasDims->outputChannels << ").";
  }

  std::array<uint32_t, 2> paddedInputSize = inputDims.getPaddedInputSize(
      params.padding.getVertical(), params.padding.getHorizontal());
  std::array<uint32_t, 2> effectiveKernelSize =
      weightDims.getEffectiveKernelSize(params.dilation.vertical,
                                        params.dilation.horizontal);
  if (paddedInputSize[0] < effectiveKernelSize[0] ||
      paddedInputSize[1] < effectiveKernelSize[1]) {
    return op->emitOpError()
           << "The effective kernel size (" << effectiveKernelSize[0] << ", "
           << effectiveKernelSize[1]
           << ") cannot be greater than the padded input size per channel ("
           << paddedInputSize[0] << ", " << paddedInputSize[1] << ").";
  }

  return mlir::success();
}

::mlir::LogicalResult verifyOutputDimensions(
    mlir::tt::ttir::Conv2dOp *op, const InputTensorDims &inputDims,
    const WeightTensorDims &weightDims,
    const std::optional<BiasTensorDims> &biasDims,
    const OutputTensorDims &outputDims, const Conv2dParams &params) {

  std::array<uint32_t, 2> paddedInputSize = inputDims.getPaddedInputSize(
      params.padding.getVertical(), params.padding.getHorizontal());
  std::array<uint32_t, 2> effectiveKernelSize =
      weightDims.getEffectiveKernelSize(params.dilation.vertical,
                                        params.dilation.horizontal);

  int32_t calculatedHOut =
      (paddedInputSize[0] - effectiveKernelSize[0]) / params.stride.vertical +
      1;
  int32_t calculatedWOut =
      (paddedInputSize[1] - effectiveKernelSize[1]) / params.stride.horizontal +
      1;

  if (!outputDims.isFlattened()) {
    // Validate each dimension of the output tensor individually since it is not
    // flattened
    if (inputDims.batchSize != outputDims.batchSize) {
      return op->emitOpError()
             << "Batch size from the input tensor (" << inputDims.batchSize
             << ") must match the first dimension of the output tensor ("
             << outputDims.batchSize << ")";
    }

    if (outputDims.outputChannels != weightDims.outputChannels) {
      return op->emitOpError()
             << "The number of output channels from the output tensor ("
             << outputDims.outputChannels
             << ") must match the number of output channels in the weight "
                "tensor ("
             << weightDims.outputChannels << "). ";
    }

    if (calculatedHOut != outputDims.outputHeight ||
        calculatedWOut != outputDims.outputWidth) {
      return op->emitOpError()
             << "The output tensor height and width dimension ("
             << outputDims.outputHeight << ", " << outputDims.outputWidth
             << ") do not match the expected dimensions (" << calculatedHOut
             << ", " << calculatedWOut << ").";
    }
  } else {
    // Validate only the last two dimensions of the output tensor since it is
    // flattened
    if (calculatedHOut * calculatedWOut * inputDims.batchSize !=
        outputDims.getFlattenedDim()) {
      return op->emitOpError()
             << "The output tensor's flattened dimension ("
             << outputDims.getFlattenedDim()
             << ") does not match the product of batch_size * output_height * "
                "output_width ("
             << inputDims.batchSize << " * " << calculatedHOut << " * "
             << calculatedWOut << " = "
             << inputDims.batchSize * calculatedHOut * calculatedWOut << ").";
    }

    if (outputDims.outputChannels != weightDims.outputChannels) {
      return op->emitOpError()
             << "The number of output channels from the output tensor ("
             << outputDims.outputChannels
             << ") must match the number of output channels in the weight "
                "tensor ("
             << weightDims.outputChannels << "). ";
    }
  }

  return mlir::success();
}

} // namespace mlir::tt::ttir::verification_utils::conv2d

namespace mlir::tt::ttir::verification_utils::conv3d {

std::tuple<InputTensorDims3d, WeightTensorDims3d,
           std::optional<BiasTensorDims3d>>
getConv3dInputDims(mlir::tt::ttir::Conv3dOp *op) {
  auto inputType = op->getInput().getType();
  int64_t batchDim = op->getBatchDim();
  int64_t depthDim = op->getDepthDim();
  int64_t heightDim = op->getHeightDim();
  int64_t widthDim = op->getWidthDim();
  int64_t channelDim = op->getChannelDim();
  InputTensorDims3d inputDims = {
      inputType.getDimSize(batchDim), inputType.getDimSize(depthDim),
      inputType.getDimSize(heightDim), inputType.getDimSize(widthDim),
      inputType.getDimSize(channelDim)};

  auto weightType = op->getWeight().getType();
  WeightTensorDims3d weightDims = {
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_OUT_CHANNEL)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_IN_CHANNEL)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_KERNEL_DEPTH)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_KERNEL_HEIGHT)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_KERNEL_WIDTH))};

  std::optional<BiasTensorDims3d> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(channelDim)};
  }

  return {inputDims, weightDims, biasDims};
}

OutputTensorDims3d getConv3dOutputDims(mlir::tt::ttir::Conv3dOp *op) {
  auto outputType = op->getResult().getType();
  OutputTensorDims3d outputDims;
  outputDims.batchSize = outputType.getDimSize(op->getBatchDim());
  outputDims.outputDepth = outputType.getDimSize(op->getDepthDim());
  outputDims.outputHeight = outputType.getDimSize(op->getHeightDim());
  outputDims.outputWidth = outputType.getDimSize(op->getWidthDim());
  outputDims.outputChannels = outputType.getDimSize(op->getChannelDim());

  return outputDims;
}

llvm::Expected<Conv3dParams> getConv3dParams(mlir::tt::ttir::Conv3dOp *op) {
  auto stride = ttmlir::utils::getTripleOfInteger<int32_t>(op->getStride());
  if (!stride) {
    return llvm::createStringError(llvm::toString(stride.takeError()) +
                                   " for stride");
  }

  auto padding = ttmlir::utils::getTripleOfInteger<int32_t>(op->getPadding());
  if (!padding) {
    return llvm::createStringError(llvm::toString(padding.takeError()) +
                                   " for padding");
  }

  return Conv3dParams{Spatial3DParam(*stride), Spatial3DParam(*padding),
                      op->getGroups(), op->getPaddingMode()};
}

mlir::LogicalResult verifyConv3dParams(mlir::tt::ttir::Conv3dOp *op,
                                       const Conv3dParams &params) {
  auto isPositive = [](int64_t v) { return v > 0; };
  auto isNonNegative = [](int64_t v) { return v >= 0; };

  if (!isPositive(params.stride.depth) || !isPositive(params.stride.vertical) ||
      !isPositive(params.stride.horizontal)) {
    return op->emitOpError("Stride attribute values must be > 0.");
  }

  if (!isNonNegative(params.padding.depth) ||
      !isNonNegative(params.padding.vertical) ||
      !isNonNegative(params.padding.horizontal)) {
    return op->emitOpError("Padding attribute values must be >= 0.");
  }

  if (params.padding_mode != "zeros" && params.padding_mode != "replicate") {
    return op->emitOpError(
        "padding_mode must be either 'zeros' or 'replicate'");
  }

  return mlir::success();
}

::mlir::LogicalResult
verifyConv3dInputDims(mlir::tt::ttir::Conv3dOp *op,
                      const InputTensorDims3d &inputDims,
                      const WeightTensorDims3d &weightDims,
                      const std::optional<BiasTensorDims3d> &biasDims,
                      const Conv3dParams &params) {

  if (inputDims.inputChannels % params.groups != 0) {
    return op->emitOpError()
           << "The number of input channels from the input tensor ("
           << inputDims.inputChannels
           << ") is not divisible by the number of groups (" << params.groups
           << ").";
  }

  if (weightDims.outputChannels % params.groups != 0) {
    return op->emitOpError()
           << "The number of output channels from the weight tensor ("
           << weightDims.outputChannels
           << ") is not divisible by the number of groups (" << params.groups
           << ").";
  }

  if (inputDims.inputChannels / params.groups != weightDims.inChannPerGroup) {
    return op->emitOpError()
           << "The number of input channels per group ("
           << inputDims.inputChannels
           << ") must match the number of input channels in the weight tensor ("
           << weightDims.inChannPerGroup << ").";
  }

  if (biasDims && biasDims->outputChannels != weightDims.outputChannels) {
    return op->emitOpError()
           << "The number of output channels from the weight tensor ("
           << weightDims.outputChannels
           << ") must match the number of output channels in the bias tensor ("
           << biasDims->outputChannels << ").";
  }

  // Verify that kernel size doesn't exceed padded input size
  std::array<uint32_t, 3> paddedInputSize = inputDims.getPaddedInputSize(
      2 * params.padding.depth, 2 * params.padding.vertical,
      2 * params.padding.horizontal);
  if (paddedInputSize[0] < weightDims.kernelDepth ||
      paddedInputSize[1] < weightDims.kernelHeight ||
      paddedInputSize[2] < weightDims.kernelWidth) {
    return op->emitOpError()
           << "The kernel size (" << weightDims.kernelDepth << ", "
           << weightDims.kernelHeight << ", " << weightDims.kernelWidth
           << ") cannot be greater than the padded input size per channel ("
           << paddedInputSize[0] << ", " << paddedInputSize[1] << ", "
           << paddedInputSize[2] << ").";
  }

  return mlir::success();
}

::mlir::LogicalResult verifyOutputDimensions(
    mlir::tt::ttir::Conv3dOp *op, const InputTensorDims3d &inputDims,
    const WeightTensorDims3d &weightDims,
    const std::optional<BiasTensorDims3d> &biasDims,
    const OutputTensorDims3d &outputDims, const Conv3dParams &params) {

  // Conv3d doesn't currently support dilation, so we use this formula:
  // D_out = (D_in + 2*pD - K_D) / sD + 1
  // H_out = (H_in + 2*pH - K_H) / sH + 1
  // W_out = (W_in + 2*pW - K_W) / sW + 1
  int32_t calculatedDOut = (inputDims.inputDepth + 2 * params.padding.depth -
                            weightDims.kernelDepth) /
                               params.stride.depth +
                           1;
  int32_t calculatedHOut =
      (inputDims.inputHeight + 2 * params.padding.vertical -
       weightDims.kernelHeight) /
          params.stride.vertical +
      1;
  int32_t calculatedWOut =
      (inputDims.inputWidth + 2 * params.padding.horizontal -
       weightDims.kernelWidth) /
          params.stride.horizontal +
      1;

  // Validate batch size
  if (inputDims.batchSize != outputDims.batchSize) {
    return op->emitOpError()
           << "Batch size from the input tensor (" << inputDims.batchSize
           << ") must match the first dimension of the output tensor ("
           << outputDims.batchSize << ")";
  }

  // Validate output channels
  if (outputDims.outputChannels != weightDims.outputChannels) {
    return op->emitOpError()
           << "The number of output channels from the output tensor ("
           << outputDims.outputChannels
           << ") must match the number of output channels in the weight "
              "tensor ("
           << weightDims.outputChannels << "). ";
  }

  // Validate spatial dimensions
  if (calculatedDOut != outputDims.outputDepth ||
      calculatedHOut != outputDims.outputHeight ||
      calculatedWOut != outputDims.outputWidth) {
    return op->emitOpError()
           << "The output tensor spatial dimensions (" << outputDims.outputDepth
           << ", " << outputDims.outputHeight << ", " << outputDims.outputWidth
           << ") do not match the expected dimensions (" << calculatedDOut
           << ", " << calculatedHOut << ", " << calculatedWOut << ").";
  }

  return mlir::success();
}

} // namespace mlir::tt::ttir::verification_utils::conv3d
