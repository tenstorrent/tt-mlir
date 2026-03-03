// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/VerificationUtils.h"

namespace mlir::tt::ttnn::utils::verification_utils::conv2d {

mlir::LogicalResult verifyTensorRanks(mlir::tt::ttnn::Conv2dOp *op) {
  if (op->getInput().getType().getRank() != 4) {
    return op->emitOpError("Input must be a 4D tensor");
  }

  if (op->getWeight().getType().getRank() != 4) {
    return op->emitOpError("Weight must be a 4D tensor");
  }

  if (op->getBias() && op->getBias().getType().getRank() != 4) {
    return op->emitOpError("Bias must be a 4D tensor");
  }

  if (op->getResult().getType().getRank() != 4) {
    return op->emitOpError("Output must be a 4D tensor");
  }

  return mlir::success();
}

std::tuple<InputTensorDims, WeightTensorDims, std::optional<BiasTensorDims>>
getConv2dInputDims(mlir::tt::ttnn::Conv2dOp *op) {
  InputTensorDims inputDims = {
      op->getBatchSize(), op->getInputHeight(), op->getInputWidth(),
      op->getInput().getType().getDimSize(INPUT_OUTPUT_CHANNEL)};

  WeightTensorDims weightDims = {
      op->getOutChannels(), op->getInChannels() / op->getGroups(),
      op->getKernelSize()[0], op->getKernelSize()[1]};

  std::optional<BiasTensorDims> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(BIAS_OUT_CHANNEL)};
  }

  return {inputDims, weightDims, biasDims};
}

OutputTensorDims getConv2dOutputDims(mlir::tt::ttnn::Conv2dOp *op) {
  return {op->getResult().getType().getDimSize(FLATTENED_DIM),
          op->getResult().getType().getDimSize(INPUT_OUTPUT_CHANNEL)};
}

llvm::Expected<Conv2dParams>
getAndVerifyConv2dParams(mlir::tt::ttnn::Conv2dOp *op) {
  auto formatAttr = [](llvm::ArrayRef<int32_t> vals) {
    llvm::SmallVector<std::string, 4> strings;
    for (int32_t v : vals) {
      strings.push_back(std::to_string(v));
    }
    return "(" + llvm::join(strings, ", ") + ")";
  };

  llvm::ArrayRef<int32_t> kernelSize = op->getKernelSize();
  if (kernelSize.size() != 2) {
    return llvm::createStringError(
        "Kernel size attribute must have two values, got: " +
        std::to_string(kernelSize.size()));
  }

  llvm::ArrayRef<int32_t> stride = op->getStride();
  if (stride.size() != 2) {
    return llvm::createStringError(
        "Stride attribute must have two values, got: " +
        std::to_string(stride.size()));
  }
  if (!llvm::all_of(stride, [](int32_t value) { return value >= 1; })) {
    return llvm::createStringError(
        "Stride attribute values must be greater than 0, got: " +
        formatAttr(stride));
  }

  llvm::ArrayRef<int32_t> padding = op->getPadding();
  if (padding.size() != 2 && padding.size() != 4) {
    return llvm::createStringError(
        "Padding attribute must have two or four values, got: " +
        std::to_string(padding.size()));
  }
  if (!llvm::all_of(padding, [](int32_t value) { return value >= 0; })) {
    return llvm::createStringError(
        "Padding attribute values must be greater than or equal to 0, got: " +
        formatAttr(padding));
  }

  llvm::ArrayRef<int32_t> dilation = op->getDilation();
  if (dilation.size() != 2) {
    return llvm::createStringError(
        "Dilation attribute must have two values, got: " +
        std::to_string(dilation.size()));
  }
  if (!llvm::all_of(dilation, [](int32_t value) { return value >= 1; })) {
    return llvm::createStringError(
        "Dilation attribute values must be greater than 0, got: " +
        formatAttr(dilation));
  }

  return Conv2dParams{Spatial2DParam(kernelSize), Spatial2DParam(stride),
                      Spatial4DParam(padding), Spatial2DParam(dilation),
                      op->getGroups()};
}

::mlir::LogicalResult verifyConv2dInputDims(
    mlir::tt::ttnn::Conv2dOp *op, const InputTensorDims &inputDims,
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

  llvm::SmallVector<uint32_t, 2> paddedInputSize = inputDims.getPaddedInputSize(
      params.padding.getVertical(), params.padding.getHorizontal());
  llvm::SmallVector<uint32_t, 2> effectiveKernelSize =
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
    mlir::tt::ttnn::Conv2dOp *op, const InputTensorDims &inputDims,
    const WeightTensorDims &weightDims,
    const std::optional<BiasTensorDims> &biasDims,
    const OutputTensorDims &outputDims, const Conv2dParams &params) {

  llvm::SmallVector<uint32_t, 2> paddedInputSize = inputDims.getPaddedInputSize(
      params.padding.getVertical(), params.padding.getHorizontal());
  llvm::SmallVector<uint32_t, 2> effectiveKernelSize =
      weightDims.getEffectiveKernelSize(params.dilation.vertical,
                                        params.dilation.horizontal);

  int32_t calculatedHOut =
      (paddedInputSize[0] - effectiveKernelSize[0]) / params.stride.vertical +
      1;
  int32_t calculatedWOut =
      (paddedInputSize[1] - effectiveKernelSize[1]) / params.stride.horizontal +
      1;

  // Validate only the flattened dim of the output tensor since it is always
  // flattened
  if (calculatedHOut * calculatedWOut * inputDims.batchSize !=
      outputDims.flattenedDim) {
    return op->emitOpError()
           << "The output tensor's flattened dimension ("
           << outputDims.flattenedDim
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
           << ") must match the number of output channels in the weight tensor "
              "("
           << weightDims.outputChannels << "). ";
  }

  return mlir::success();
}

} // namespace mlir::tt::ttnn::utils::verification_utils::conv2d

namespace mlir::tt::ttnn::utils::verification_utils::conv3d {

mlir::LogicalResult verifyTensorRanks(mlir::tt::ttnn::Conv3dOp *op) {
  if (op->getInput().getType().getRank() != 5) {
    return op->emitOpError("input must be a 5D tensor [N, D, H, W, C]");
  }
  if (op->getWeight().getType().getRank() != 2) {
    return op->emitOpError("weight must be a 2D tensor [kD*kH*kW*C/G, O]");
  }
  if (op->getBias() && op->getBias().getType().getRank() != 2) {
    return op->emitOpError("bias must be a 2D tensor [1, O]");
  }
  if (op->getResult().getType().getRank() != 5) {
    return op->emitOpError(
        "result must be a 5D tensor [N, D_out, H_out, W_out, O]");
  }
  return mlir::success();
}

std::tuple<InputTensorDims3d, WeightTensorDims3d,
           std::optional<BiasTensorDims3d>>
getConv3dInputDims(mlir::tt::ttnn::Conv3dOp *op) {
  InputTensorDims3d inputDims = {op->getBatchSize(), op->getInputDepth(),
                                 op->getInputHeight(), op->getInputWidth(),
                                 op->getInChannels()};

  WeightTensorDims3d weightDims = {
      op->getOutChannels(),
      op->getWeight().getType().getDimSize(WEIGHT_FLATTENED),
      op->getKernelSize()[0], op->getKernelSize()[1], op->getKernelSize()[2]};

  std::optional<BiasTensorDims3d> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(BIAS_FIRST_DIM),
                op->getBias().getType().getDimSize(BIAS_OUT_CHANNEL)};
  }

  return {inputDims, weightDims, biasDims};
}

OutputTensorDims3d getConv3dOutputDims(mlir::tt::ttnn::Conv3dOp *op) {
  return {op->getResult().getType().getDimSize(OUTPUT_BATCH),
          op->getResult().getType().getDimSize(OUTPUT_DEPTH),
          op->getResult().getType().getDimSize(OUTPUT_HEIGHT),
          op->getResult().getType().getDimSize(OUTPUT_WIDTH),
          op->getResult().getType().getDimSize(OUTPUT_CHANNEL)};
}

llvm::Expected<Conv3dParams>
getAndVerifyConv3dParams(mlir::tt::ttnn::Conv3dOp *op) {
  auto formatAttr = [](llvm::ArrayRef<int32_t> vals) {
    llvm::SmallVector<std::string, 3> strings;
    for (int32_t v : vals) {
      strings.push_back(std::to_string(v));
    }
    return "(" + llvm::join(strings, ", ") + ")";
  };

  llvm::ArrayRef<int32_t> kernelSize = op->getKernelSize();
  if (kernelSize.size() != 3) {
    return llvm::createStringError("kernel_size must have 3 values, got: " +
                                   std::to_string(kernelSize.size()));
  }

  if (!llvm::all_of(kernelSize, [](int32_t v) { return v >= 1; })) {
    return llvm::createStringError("kernel_size values must be >= 1, got: " +
                                   formatAttr(kernelSize));
  }

  llvm::ArrayRef<int32_t> stride = op->getStride();
  if (stride.size() != 3) {
    return llvm::createStringError("stride must have 3 values, got: " +
                                   std::to_string(stride.size()));
  }
  if (!llvm::all_of(stride, [](int32_t v) { return v >= 1; })) {
    return llvm::createStringError("stride values must be > 0, got: " +
                                   formatAttr(stride));
  }

  llvm::ArrayRef<int32_t> padding = op->getPadding();
  if (padding.size() != 3) {
    return llvm::createStringError("padding must have 3 values, got: " +
                                   std::to_string(padding.size()));
  }
  if (!llvm::all_of(padding, [](int32_t v) { return v >= 0; })) {
    return llvm::createStringError("padding values must be >= 0, got: " +
                                   formatAttr(padding));
  }

  llvm::StringRef paddingMode = op->getPaddingMode();
  if (paddingMode != "zeros" && paddingMode != "replicate") {
    return llvm::createStringError(
        "padding_mode must be 'zeros' or 'replicate', got: " +
        paddingMode.str());
  }

  if (op->getGroups() < 1) {
    return llvm::createStringError("groups must be >= 1, got: " +
                                   std::to_string(op->getGroups()));
  }

  return Conv3dParams{Spatial3DParam(kernelSize), Spatial3DParam(stride),
                      Spatial3DParam(padding), op->getGroups(), paddingMode};
}

::mlir::LogicalResult
verifyConv3dInputDims(mlir::tt::ttnn::Conv3dOp *op,
                      const InputTensorDims3d &inputDims,
                      const WeightTensorDims3d &weightDims,
                      const std::optional<BiasTensorDims3d> &biasDims,
                      const Conv3dParams &params) {

  if (inputDims.inputChannels % params.groups != 0) {
    return op->emitOpError()
           << "in_channels (" << inputDims.inputChannels
           << ") must be divisible by groups (" << params.groups << ")";
  }

  if (weightDims.outputChannels % params.groups != 0) {
    return op->emitOpError()
           << "out_channels (" << weightDims.outputChannels
           << ") must be divisible by groups (" << params.groups << ")";
  }

  int64_t expectedFlattenedDim =
      (inputDims.inputChannels / params.groups) * params.kernelSize.depth *
      params.kernelSize.vertical * params.kernelSize.horizontal;

  if (expectedFlattenedDim != weightDims.flattenedKernelChannels) {
    return op->emitOpError() << "weight flattened dimension ("
                             << weightDims.flattenedKernelChannels
                             << ") must equal kD*kH*kW*C_in/groups ("
                             << expectedFlattenedDim << ")";
  }

  if (biasDims) {
    if (biasDims->outputChannels != weightDims.outputChannels) {
      return op->emitOpError()
             << "bias output channels (" << biasDims->outputChannels
             << ") must match weight output channels ("
             << weightDims.outputChannels << ")";
    }
    if (biasDims->firstDim != 1) {
      return op->emitOpError("bias first dimension must be 1, got ")
             << biasDims->firstDim;
    }
  }

  return mlir::success();
}

::mlir::LogicalResult verifyConv3dOutputDims(
    mlir::tt::ttnn::Conv3dOp *op, const InputTensorDims3d &inputDims,
    const WeightTensorDims3d &weightDims, const OutputTensorDims3d &outputDims,
    const Conv3dParams &params) {

  int32_t calculatedDOut = (inputDims.inputDepth + 2 * params.padding.depth -
                            params.kernelSize.depth) /
                               params.stride.depth +
                           1;

  int32_t calculatedHOut =
      (inputDims.inputHeight + 2 * params.padding.vertical -
       params.kernelSize.vertical) /
          params.stride.vertical +
      1;

  int32_t calculatedWOut =
      (inputDims.inputWidth + 2 * params.padding.horizontal -
       params.kernelSize.horizontal) /
          params.stride.horizontal +
      1;

  if (outputDims.batchSize != inputDims.batchSize) {
    return op->emitOpError()
           << "output batch size (" << outputDims.batchSize
           << ") must match input batch size (" << inputDims.batchSize << ")";
  }

  if (outputDims.outputDepth != calculatedDOut) {
    return op->emitOpError()
           << "output depth (" << outputDims.outputDepth
           << ") does not match calculated depth (" << calculatedDOut << ")";
  }

  if (outputDims.outputHeight != calculatedHOut) {
    return op->emitOpError()
           << "output height (" << outputDims.outputHeight
           << ") does not match calculated height (" << calculatedHOut << ")";
  }

  if (outputDims.outputWidth != calculatedWOut) {
    return op->emitOpError()
           << "output width (" << outputDims.outputWidth
           << ") does not match calculated width (" << calculatedWOut << ")";
  }

  if (outputDims.outputChannels != weightDims.outputChannels) {
    return op->emitOpError() << "output channels (" << outputDims.outputChannels
                             << ") must match weight output channels ("
                             << weightDims.outputChannels << ")";
  }

  return mlir::success();
}

} // namespace mlir::tt::ttnn::utils::verification_utils::conv3d
