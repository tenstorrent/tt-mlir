// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_VERIFICATIONUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_VERIFICATIONUTILS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn::utils::verification_utils {
namespace conv2d_verification {
// Input and output tensors are always flattened
enum InputOutputDim : unsigned { FLATTENED_DIM = 2, INPUT_OUTPUT_CHANNEL = 3 };

enum WeightDim : unsigned {
  WEIGHT_OUT_CHANNEL = 0,
  WEIGHT_IN_CHANNEL = 1,
  WEIGHT_KERNEL_HEIGHT = 2,
  WEIGHT_KERNEL_WIDTH = 3
};

enum BiasDim : unsigned { BIAS_OUT_CHANNEL = 3 };

struct InputTensorDims {
  int64_t batchSize;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t inputChannels;

  llvm::SmallVector<uint32_t, 2>
  getPaddedInputSize(int64_t verticalPadding, int64_t horizontalPadding) const {
    return {static_cast<uint32_t>(inputHeight + verticalPadding),
            static_cast<uint32_t>(inputWidth + horizontalPadding)};
  }
};

struct WeightTensorDims {
  int64_t outputChannels;
  int64_t inChannPerGroup;
  int64_t kernelHeight;
  int64_t kernelWidth;

  llvm::SmallVector<uint32_t, 2>
  getEffectiveKernelSize(int64_t verticalDilation,
                         int64_t horizontalDilation) const {
    int64_t effectiveKernelHeight = verticalDilation * (kernelHeight - 1) + 1;
    int64_t effectiveKernelWidth = horizontalDilation * (kernelWidth - 1) + 1;
    return {static_cast<uint32_t>(effectiveKernelHeight),
            static_cast<uint32_t>(effectiveKernelWidth)};
  }
};

struct BiasTensorDims {
  int64_t outputChannels;
};

struct OutputTensorDims {
  int64_t flattenedDim;
  int64_t outputChannels;
};

struct Spatial2DParam {
  int64_t vertical, horizontal;

  Spatial2DParam(llvm::ArrayRef<int32_t> p)
      : vertical(p[0]), horizontal(p[1]) {}
};

struct Spatial4DParam {
  int64_t top, left, bottom, right;

  Spatial4DParam(llvm::ArrayRef<int32_t> p) {
    assert(p.size() == 2 || p.size() == 4);

    if (p.size() == 2) {
      // [vertical, horizontal]
      top = bottom = p[0];
      left = right = p[1];
    } else {
      // [top, bottom, left, right]
      top = p[0];
      bottom = p[1];
      left = p[2];
      right = p[3];
    }
  }

  int64_t getVertical() const { return top + bottom; }
  int64_t getHorizontal() const { return left + right; }
};

struct Conv2dParams {
  Spatial2DParam kernelSize;
  Spatial2DParam stride;
  Spatial4DParam padding;
  Spatial2DParam dilation;
  int64_t groups;
};

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

} // namespace conv2d_verification

} // namespace mlir::tt::ttnn::utils::verification_utils

#endif
