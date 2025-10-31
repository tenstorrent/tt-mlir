// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/MathExtras.h"

#include <array>

namespace mlir::tt::ttir::verification_utils {

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

enum class InputDim : unsigned {
  INPUT_BATCH = 0,
  INPUT_HEIGHT = 1,
  INPUT_WIDTH = 2,
  INPUT_CHANNEL = 3
};

enum class OutputDim : unsigned {
  OUTPUT_BATCH = 0,
  OUTPUT_HEIGHT = 1,
  OUTPUT_WIDTH = 2,
  OUTPUT_CHANNEL = 3
};

// If the input and output tensors are flattened, this is the dimension upon
// which they are flattened.
constexpr unsigned int FLATTENED_DIM = 2;

struct InputTensorDims {
  int64_t batchSize;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t inputChannels;

  std::array<uint32_t, 2> getPaddedInputSize(int64_t verticalPadding,
                                             int64_t horizontalPadding) const {
    return {static_cast<uint32_t>(inputHeight + verticalPadding),
            static_cast<uint32_t>(inputWidth + horizontalPadding)};
  }
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

enum class WeightDim : unsigned {
  WEIGHT_OUT_CHANNEL = 0,
  WEIGHT_IN_CHANNEL = 1,
  WEIGHT_KERNEL_HEIGHT = 2,
  WEIGHT_KERNEL_WIDTH = 3
};

enum class BiasDim : unsigned { BIAS_OUT_CHANNEL = 3 };

struct WeightTensorDims {
  int64_t outputChannels;
  int64_t inChannPerGroup;
  int64_t kernelHeight;
  int64_t kernelWidth;

  std::array<uint32_t, 2>
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

struct Conv2dParams {
  Spatial2DParam stride;
  Spatial4DParam padding;
  Spatial2DParam dilation;
  int64_t groups;
};

template <typename Op, bool HasWeightAndBias = false>
mlir::LogicalResult verifyTensorRanks(Op *op) {
  if (op->getInput().getType().getRank() != 4) {
    return op->emitOpError("input must be a 4D tensor");
  }

  if constexpr (HasWeightAndBias) {
    if (op->getWeight().getType().getRank() != 4) {
      return op->emitOpError("weight must be a 4D tensor");
    }

    if (op->getBias() && op->getBias().getType().getRank() != 4) {
      return op->emitOpError("bias must be a 4D tensor");
    }
  }

  if (op->getResult().getType().getRank() != 4) {
    return op->emitOpError("output must be a 4D tensor");
  }

  return mlir::success();
}

inline std::tuple<InputTensorDims, WeightTensorDims,
                  std::optional<BiasTensorDims>>
getConv2dInputDims(mlir::tt::ttir::Conv2dOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfoAttr();
  InputTensorDims inputDims;
  auto inputType = op->getInput().getType();
  if (flatInfo) {
    inputDims = {
        flatInfo.getBatchSize(), flatInfo.getInputHeight(),
        flatInfo.getInputWidth(),
        inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_CHANNEL))};
  } else {
    inputDims = {
        inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_BATCH)),
        inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_HEIGHT)),
        inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_WIDTH)),
        inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_CHANNEL))};
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
    biasDims = {op->getBias().getType().getDimSize(
        llvm::to_underlying(BiasDim::BIAS_OUT_CHANNEL))};
  }

  return {inputDims, weightDims, biasDims};
}

inline OutputTensorDims getConv2dOutputDims(mlir::tt::ttir::Conv2dOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfoAttr();
  OutputTensorDims outputDims;
  auto outputType = op->getOutput().getType();
  if (flatInfo) {
    outputDims.flattenedDim = outputType.getDimSize(FLATTENED_DIM);
    outputDims.outputChannels =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_CHANNEL));
  } else {
    outputDims.batchSize =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_BATCH));
    outputDims.outputHeight =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_HEIGHT));
    outputDims.outputWidth =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_WIDTH));
    outputDims.outputChannels =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_CHANNEL));
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

inline ::mlir::LogicalResult verifyOutputDimensions(
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

struct Pool2dParams {
  Spatial2DParam kernel;
  Spatial2DParam stride;
  Spatial2DParam dilation;
  Spatial4DParam padding;
  bool ceilMode;

  std::array<uint32_t, 2> getEffectiveKernelSize() const {
    int64_t effectiveKernelHeight =
        dilation.vertical * (kernel.vertical - 1) + 1;
    int64_t effectiveKernelWidth =
        dilation.horizontal * (kernel.horizontal - 1) + 1;
    return {static_cast<uint32_t>(effectiveKernelHeight),
            static_cast<uint32_t>(effectiveKernelWidth)};
  }
};

template <typename PoolOp>
mlir::LogicalResult verifyFlattenedCompatInfo(PoolOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfo();
  if (flatInfo) {
    int64_t batchSize = flatInfo.getBatchSize();
    int64_t inputHeight = flatInfo.getInputHeight();
    int64_t inputWidth = flatInfo.getInputWidth();
    int64_t expectedSize = batchSize * inputHeight * inputWidth;
    int64_t actualSize = op->getInput().getType().getDimSize(FLATTENED_DIM);

    if (expectedSize != actualSize) {
      return op->emitOpError()
             << "the input tensor's flattened dimension (" << actualSize
             << ") does not match the product of batch_size * input_height * "
                "input_width from FlattenedCompatInfo ("
             << flatInfo.getBatchSize() << " * " << flatInfo.getInputHeight()
             << " * " << flatInfo.getInputWidth() << " = " << expectedSize
             << ")";
    }
  }
  return mlir::success();
}

template <typename PoolOp>
InputTensorDims getPool2dInputDims(PoolOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfo();
  auto inputType = op->getInput().getType();
  if (flatInfo) {
    return {flatInfo.getBatchSize(), flatInfo.getInputHeight(),
            flatInfo.getInputWidth(),
            inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_CHANNEL))};
  }
  return {inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_BATCH)),
          inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_HEIGHT)),
          inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_WIDTH)),
          inputType.getDimSize(llvm::to_underlying(InputDim::INPUT_CHANNEL))};
}

template <typename PoolOp>
OutputTensorDims getPool2dOutputDims(PoolOp *op) {
  mlir::tt::ttir::FlattenedCompatInfoAttr flatInfo =
      op->getFlattenedCompatInfo();
  OutputTensorDims outputDims;
  auto outputType = op->getResult().getType();
  if (flatInfo) {
    outputDims.flattenedDim = outputType.getDimSize(FLATTENED_DIM);
    outputDims.outputChannels =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_CHANNEL));
  } else {
    outputDims.batchSize =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_BATCH));
    outputDims.outputHeight =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_HEIGHT));
    outputDims.outputWidth =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_WIDTH));
    outputDims.outputChannels =
        outputType.getDimSize(llvm::to_underlying(OutputDim::OUTPUT_CHANNEL));
  }

  return outputDims;
}

template <typename PoolOp>
llvm::Expected<Pool2dParams> getPool2dParams(PoolOp *op) {
  auto kernel = ttmlir::utils::getPairOfInteger<int32_t>(op->getKernel());
  if (!kernel) {
    return llvm::createStringError(llvm::toString(kernel.takeError()) +
                                   " for kernel attribute");
  }

  auto stride = ttmlir::utils::getPairOfInteger<int32_t>(op->getStride());
  if (!stride) {
    return llvm::createStringError(llvm::toString(stride.takeError()) +
                                   " for stride attribute");
  }

  auto dilation = ttmlir::utils::getPairOfInteger<int32_t>(op->getDilation());
  if (!dilation) {
    return llvm::createStringError(llvm::toString(dilation.takeError()) +
                                   " for dilation attribute");
  }

  auto padding =
      ttmlir::utils::getQuadrupleOfInteger<int32_t>(op->getPadding());
  if (!padding) {
    return llvm::createStringError(llvm::toString(padding.takeError()) +
                                   " for padding attribute");
  }

  bool ceilMode = op->getCeilMode();

  return Pool2dParams{Spatial2DParam(*kernel), Spatial2DParam(*stride),
                      Spatial2DParam(*dilation), Spatial4DParam(*padding),
                      ceilMode};
}

template <typename PoolOp>
mlir::LogicalResult verifyPool2dInputDims(PoolOp *op,
                                          const InputTensorDims &inputDims,
                                          const Pool2dParams &params) {
  std::array<uint32_t, 2> paddedInputSize = inputDims.getPaddedInputSize(
      params.padding.getVertical(), params.padding.getHorizontal());
  std::array<uint32_t, 2> effectiveKernelSize = params.getEffectiveKernelSize();
  if (paddedInputSize[0] < effectiveKernelSize[0] ||
      paddedInputSize[1] < effectiveKernelSize[1]) {
    return op->emitOpError()
           << "effective kernel size (" << effectiveKernelSize[0] << ", "
           << effectiveKernelSize[1]
           << ") cannot be greater than the padded input size per channel ("
           << paddedInputSize[0] << ", " << paddedInputSize[1] << ")";
  }

  return mlir::success();
}

template <typename PoolOp>
mlir::LogicalResult verifyPool2dOutputDims(PoolOp *op,
                                           const InputTensorDims &inputDims,
                                           const OutputTensorDims &outputDims,
                                           const Pool2dParams &params) {

  // Calculate expected output dimensions.
  int32_t paddedHeight = inputDims.inputHeight + params.padding.getVertical();
  int32_t paddedWidth = inputDims.inputWidth + params.padding.getHorizontal();

  int32_t effectiveKernelHeight = params.getEffectiveKernelSize()[0];
  int32_t effectiveKernelWidth = params.getEffectiveKernelSize()[1];

  int32_t calculatedHOut, calculatedWOut;
  // Adjust for ceil/floor mode. If ceilMode is true, we use ceiling division;
  // otherwise, we use floor division.
  if (params.ceilMode) {
    // Ceiling mode: use ceiling division.
    calculatedHOut = llvm::divideCeil(paddedHeight - effectiveKernelHeight,
                                      params.stride.vertical) +
                     1;
    calculatedWOut = llvm::divideCeil(paddedWidth - effectiveKernelWidth,
                                      params.stride.horizontal) +
                     1;

    // Adjust the output shape if the last kernel position is in the padding
    // region
    if ((calculatedHOut - 1) * params.stride.vertical >=
        inputDims.inputHeight + params.padding.top) {
      calculatedHOut--;
    }
    if ((calculatedWOut - 1) * params.stride.horizontal >=
        inputDims.inputWidth + params.padding.left) {
      calculatedWOut--;
    }
  } else {
    // Floor mode: use floor division (standard integer division).
    calculatedHOut =
        llvm::divideFloorSigned(paddedHeight - effectiveKernelHeight,
                                params.stride.vertical) +
        1;
    calculatedWOut = llvm::divideFloorSigned(paddedWidth - effectiveKernelWidth,
                                             params.stride.horizontal) +
                     1;
  }

  if (!outputDims.isFlattened()) {
    // Validate each dimension of the output tensor individually since it is not
    // flattened.
    if (inputDims.batchSize != outputDims.batchSize) {
      return op->emitOpError()
             << "batch size from the input tensor (" << inputDims.batchSize
             << ") must match the first dimension of the output tensor ("
             << outputDims.batchSize << ")";
    }

    if (outputDims.outputChannels != inputDims.inputChannels) {
      return op->emitOpError()
             << "number of output channels from the output tensor ("
             << outputDims.outputChannels
             << ") must match the number of input channels ("
             << inputDims.inputChannels << ")";
    }

    if (calculatedHOut != outputDims.outputHeight ||
        calculatedWOut != outputDims.outputWidth) {
      return op->emitOpError()
             << "output tensor height and width dimension ("
             << outputDims.outputHeight << ", " << outputDims.outputWidth
             << ") do not match the expected dimensions (" << calculatedHOut
             << ", " << calculatedWOut << ")";
    }
  } else {
    // Validate only the last two dimensions of the output tensor since it is
    // flattened.
    if (outputDims.outputChannels != inputDims.inputChannels) {
      return op->emitOpError()
             << "number of output channels from the output tensor ("
             << outputDims.outputChannels
             << ") must match the number of input channels ("
             << inputDims.inputChannels << ")";
    }

    if (calculatedHOut * calculatedWOut * inputDims.batchSize !=
        outputDims.getFlattenedDim()) {
      return op->emitOpError()
             << "output tensor's flattened dimension ("
             << outputDims.getFlattenedDim()
             << ") does not match the product of batch_size * output_height * "
                "output_width ("
             << inputDims.batchSize << " * " << calculatedHOut << " * "
             << calculatedWOut << " = "
             << inputDims.batchSize * calculatedHOut * calculatedWOut << ")";
    }
  }

  return mlir::success();
}

template <typename PoolOp>
mlir::LogicalResult verifyPool2dParams(PoolOp *op, const Pool2dParams &params) {
  auto isPositive = [](int64_t v) { return v > 0; };
  auto isNonNegative = [](int64_t v) { return v >= 0; };

  if (!isPositive(params.kernel.vertical) ||
      !isPositive(params.kernel.horizontal)) {
    return op->emitOpError("kernel size attribute values must be > 0");
  }

  if (!isPositive(params.stride.vertical) ||
      !isPositive(params.stride.horizontal)) {
    return op->emitOpError("stride attribute values must be > 0");
  }

  if (!isPositive(params.dilation.vertical) ||
      !isPositive(params.dilation.horizontal)) {
    return op->emitOpError("dilation attribute values must be > 0");
  }

  if (!isNonNegative(params.padding.top) ||
      !isNonNegative(params.padding.left) ||
      !isNonNegative(params.padding.bottom) ||
      !isNonNegative(params.padding.right)) {
    return op->emitOpError("padding attribute values must be >= 0");
  }

  return mlir::success();
}

} // namespace mlir::tt::ttir::verification_utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H
