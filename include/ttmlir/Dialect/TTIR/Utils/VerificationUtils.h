// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir::verification_utils {
namespace conv2d_verification {
enum InputDim : unsigned {
  INPUT_BATCH = 0,
  INPUT_HEIGHT = 1,
  INPUT_WIDTH = 2,
  INPUT_CHANNEL = 3
};

enum OutputDim : unsigned {
  OUTPUT_BATCH = 0,
  OUTPUT_HEIGHT = 1,
  OUTPUT_WIDTH = 2,
  OUTPUT_CHANNEL = 3
};

// If the input and output tensors are flattened, this is the dimension upon
// which they are flattened.
constexpr unsigned int FLATTENED_DIM = 2;

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
  int64_t batchSize;
  int64_t outputHeight;
  int64_t outputWidth;
  int64_t outputChannels;
  std::optional<int64_t> flattenedDim;

  bool isFlattened() const { return flattenedDim.has_value(); }

  int64_t getFlattenedDim() const { return flattenedDim.value(); }
};

struct Spatial2DParam {
  int64_t vertical, horizontal;

  Spatial2DParam(std::pair<int64_t, int64_t> p)
      : vertical(p.first), horizontal(p.second) {}
};

struct Spatial4DParam {
  int64_t top, left, bottom, right;

  Spatial4DParam(std::tuple<int64_t, int64_t, int64_t, int64_t> p)
      : top(std::get<0>(p)), left(std::get<1>(p)), bottom(std::get<2>(p)),
        right(std::get<3>(p)) {}

  int64_t getVertical() const { return top + bottom; }
  int64_t getHorizontal() const { return left + right; }
};

struct Conv2dParams {
  Spatial2DParam stride;
  Spatial4DParam padding;
  Spatial2DParam dilation;
  int64_t groups;
};

inline mlir::LogicalResult verifyTensorRanks(mlir::tt::ttir::Conv2dOp *op) {
  if (op->getInput().getType().getRank() != 4) {
    return op->emitOpError("Input must be a 4D tensor");
  }

  if (op->getWeight().getType().getRank() != 4) {
    return op->emitOpError("Weight must be a 4D tensor");
  }

  if (op->getBias() && op->getBias().getType().getRank() != 4) {
    return op->emitOpError("Bias must be a 4D tensor");
  }

  if (op->getOutput().getType().getRank() != 4) {
    return op->emitOpError("Output must be a 4D tensor");
  }

  return mlir::success();
}

inline std::tuple<InputTensorDims, WeightTensorDims,
                  std::optional<BiasTensorDims>>
getConv2dInputDims(mlir::tt::ttir::Conv2dOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfoAttr();
  InputTensorDims inputDims;
  if (flatInfo) {
    inputDims = {flatInfo.getBatchSize(), flatInfo.getInputHeight(),
                 flatInfo.getInputWidth(),
                 op->getInput().getType().getDimSize(INPUT_CHANNEL)};
  } else {
    inputDims = {op->getInput().getType().getDimSize(INPUT_BATCH),
                 op->getInput().getType().getDimSize(INPUT_HEIGHT),
                 op->getInput().getType().getDimSize(INPUT_WIDTH),
                 op->getInput().getType().getDimSize(INPUT_CHANNEL)};
  }

  WeightTensorDims weightDims = {
      op->getWeight().getType().getDimSize(WEIGHT_OUT_CHANNEL),
      op->getWeight().getType().getDimSize(WEIGHT_IN_CHANNEL),
      op->getWeight().getType().getDimSize(WEIGHT_KERNEL_HEIGHT),
      op->getWeight().getType().getDimSize(WEIGHT_KERNEL_WIDTH)};

  std::optional<BiasTensorDims> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(BIAS_OUT_CHANNEL)};
  }

  return {inputDims, weightDims, biasDims};
}

inline OutputTensorDims getConv2dOutputDims(mlir::tt::ttir::Conv2dOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfoAttr();
  OutputTensorDims outputDims;
  if (flatInfo) {
    outputDims.flattenedDim =
        op->getOutput().getType().getDimSize(FLATTENED_DIM);
    outputDims.outputChannels =
        op->getOutput().getType().getDimSize(OUTPUT_CHANNEL);
  } else {
    outputDims.batchSize = op->getOutput().getType().getDimSize(OUTPUT_BATCH);
    outputDims.outputHeight =
        op->getOutput().getType().getDimSize(OUTPUT_HEIGHT);
    outputDims.outputWidth = op->getOutput().getType().getDimSize(OUTPUT_WIDTH);
    outputDims.outputChannels =
        op->getOutput().getType().getDimSize(OUTPUT_CHANNEL);
  }

  return outputDims;
}

inline llvm::Expected<Conv2dParams>
getConv2dParams(mlir::tt::ttir::Conv2dOp *op) {
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

inline mlir::LogicalResult verifyConv2dParams(mlir::tt::ttir::Conv2dOp *op,
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

inline ::mlir::LogicalResult verifyConv2dInputDims(
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

inline ::mlir::LogicalResult verifyOutputDimensions(
    mlir::tt::ttir::Conv2dOp *op, const InputTensorDims &inputDims,
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

} // namespace conv2d_verification

} // namespace mlir::tt::ttir::verification_utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H
