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
  auto outputType = op->getType();
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

// Conv3d support structures
struct Spatial3DParam {
  int64_t depth, vertical, horizontal;

  Spatial3DParam(std::tuple<int64_t, int64_t, int64_t> p)
      : depth(std::get<0>(p)), vertical(std::get<1>(p)),
        horizontal(std::get<2>(p)) {}
};

enum class InputDim3d : unsigned {
  INPUT_BATCH = 0,
  INPUT_DEPTH = 1,
  INPUT_HEIGHT = 2,
  INPUT_WIDTH = 3,
  INPUT_CHANNEL = 4
};

enum class OutputDim3d : unsigned {
  OUTPUT_BATCH = 0,
  OUTPUT_DEPTH = 1,
  OUTPUT_HEIGHT = 2,
  OUTPUT_WIDTH = 3,
  OUTPUT_CHANNEL = 4
};

enum class WeightDim3d : unsigned {
  WEIGHT_OUT_CHANNEL = 0,
  WEIGHT_IN_CHANNEL = 1,
  WEIGHT_KERNEL_DEPTH = 2,
  WEIGHT_KERNEL_HEIGHT = 3,
  WEIGHT_KERNEL_WIDTH = 4
};

enum class BiasDim3d : unsigned { BIAS_OUT_CHANNEL = 0 };

struct InputTensorDims3d {
  int64_t batchSize;
  int64_t inputDepth;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t inputChannels;

  std::array<uint32_t, 3> getPaddedInputSize(int64_t depthPadding,
                                             int64_t verticalPadding,
                                             int64_t horizontalPadding) const {
    return {static_cast<uint32_t>(inputDepth + depthPadding),
            static_cast<uint32_t>(inputHeight + verticalPadding),
            static_cast<uint32_t>(inputWidth + horizontalPadding)};
  }
};

struct WeightTensorDims3d {
  int64_t outputChannels;
  int64_t inChannPerGroup;
  int64_t kernelDepth;
  int64_t kernelHeight;
  int64_t kernelWidth;
};

struct BiasTensorDims3d {
  int64_t outputChannels;
};

struct OutputTensorDims3d {
  int64_t batchSize;
  int64_t outputDepth;
  int64_t outputHeight;
  int64_t outputWidth;
  int64_t outputChannels;
};

struct Conv3dParams {
  Spatial3DParam stride;
  Spatial3DParam padding;
  int64_t groups;
  llvm::StringRef padding_mode;
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


inline std::tuple<InputTensorDims3d, WeightTensorDims3d,
                  std::optional<BiasTensorDims3d>>
getConv3dInputDims(mlir::tt::ttir::Conv3dOp *op) {
  auto inputType = op->getInput().getType();
  InputTensorDims3d inputDims = {
      inputType.getDimSize(llvm::to_underlying(InputDim3d::INPUT_BATCH)),
      inputType.getDimSize(llvm::to_underlying(InputDim3d::INPUT_DEPTH)),
      inputType.getDimSize(llvm::to_underlying(InputDim3d::INPUT_HEIGHT)),
      inputType.getDimSize(llvm::to_underlying(InputDim3d::INPUT_WIDTH)),
      inputType.getDimSize(llvm::to_underlying(InputDim3d::INPUT_CHANNEL))};

  auto weightType = op->getWeight().getType();
  WeightTensorDims3d weightDims = {
      weightType.getDimSize(llvm::to_underlying(WeightDim3d::WEIGHT_OUT_CHANNEL)),
      weightType.getDimSize(llvm::to_underlying(WeightDim3d::WEIGHT_IN_CHANNEL)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_KERNEL_DEPTH)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_KERNEL_HEIGHT)),
      weightType.getDimSize(
          llvm::to_underlying(WeightDim3d::WEIGHT_KERNEL_WIDTH))};

  std::optional<BiasTensorDims3d> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(
        llvm::to_underlying(BiasDim3d::BIAS_OUT_CHANNEL))};
  }

  return {inputDims, weightDims, biasDims};
}

inline OutputTensorDims3d getConv3dOutputDims(mlir::tt::ttir::Conv3dOp *op) {
  auto outputType = op->getOutput().getType();
  OutputTensorDims3d outputDims;
  outputDims.batchSize =
      outputType.getDimSize(llvm::to_underlying(OutputDim3d::OUTPUT_BATCH));
  outputDims.outputDepth =
      outputType.getDimSize(llvm::to_underlying(OutputDim3d::OUTPUT_DEPTH));
  outputDims.outputHeight =
      outputType.getDimSize(llvm::to_underlying(OutputDim3d::OUTPUT_HEIGHT));
  outputDims.outputWidth =
      outputType.getDimSize(llvm::to_underlying(OutputDim3d::OUTPUT_WIDTH));
  outputDims.outputChannels =
      outputType.getDimSize(llvm::to_underlying(OutputDim3d::OUTPUT_CHANNEL));

  return outputDims;
}

inline llvm::Expected<Conv3dParams>
getConv3dParams(mlir::tt::ttir::Conv3dOp *op) {
  auto stride = ttmlir::utils::getTripleOfInteger<int32_t>(
      op->getStride());
  if (!stride) {
    return llvm::createStringError(llvm::toString(stride.takeError()) +
                                   " for stride");
  }

  auto padding = ttmlir::utils::getTripleOfInteger<int32_t>(
      op->getPadding());
  if (!padding) {
    return llvm::createStringError(llvm::toString(padding.takeError()) +
                                   " for padding");
  }

  return Conv3dParams{Spatial3DParam(*stride), Spatial3DParam(*padding),
                      op->getGroups(), op->getPaddingMode()};
}

inline mlir::LogicalResult verifyConv3dParams(mlir::tt::ttir::Conv3dOp *op,
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

inline ::mlir::LogicalResult verifyConv3dInputDims(
    mlir::tt::ttir::Conv3dOp *op, const InputTensorDims3d &inputDims,
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

inline ::mlir::LogicalResult
verifyOutputDimensions(mlir::tt::ttir::Conv3dOp *op,
                       const InputTensorDims3d &inputDims,
                       const WeightTensorDims3d &weightDims,
                       const std::optional<BiasTensorDims3d> &biasDims,
                       const OutputTensorDims3d &outputDims,
                       const Conv3dParams &params) {

  // Conv3d doesn't currently support dilation, so we use this formula:
  // D_out = (D_in + 2*pD - K_D) / sD + 1
  // H_out = (H_in + 2*pH - K_H) / sH + 1
  // W_out = (W_in + 2*pW - K_W) / sW + 1
  int32_t calculatedDOut = (inputDims.inputDepth + 2 * params.padding.depth -
                            weightDims.kernelDepth) /
                               params.stride.depth +
                           1;
  int32_t calculatedHOut = (inputDims.inputHeight + 2 * params.padding.vertical -
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

} // namespace mlir::tt::ttir::verification_utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_VERIFICATIONUTILS_H
