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

} // namespace conv2d_verification

namespace conv3d_verification {
// Conv3d input/output tensors use 5D shapes, weight/bias use 2D shapes
enum InputDim : unsigned {
  INPUT_BATCH = 0,
  INPUT_DEPTH = 1,
  INPUT_HEIGHT = 2,
  INPUT_WIDTH = 3,
  INPUT_CHANNEL = 4
};

enum OutputDim : unsigned {
  OUTPUT_BATCH = 0,
  OUTPUT_DEPTH = 1,
  OUTPUT_HEIGHT = 2,
  OUTPUT_WIDTH = 3,
  OUTPUT_CHANNEL = 4
};

// Weight is 2D: [kD*kH*kW*C_in, O]
enum WeightDim : unsigned {
  WEIGHT_FLATTENED = 0,  // kD*kH*kW*C_in (patch_size)
  WEIGHT_OUT_CHANNEL = 1 // O (output channels)
};

// Bias is 2D: [32, O]
enum BiasDim : unsigned {
  BIAS_TILE_HEIGHT = 0,  // Must be 32 (tile height)
  BIAS_OUT_CHANNEL = 1   // Must be O (output channels)
};

struct InputTensorDims3d {
  int64_t batchSize;
  int64_t inputDepth;
  int64_t inputHeight;
  int64_t inputWidth;
  int64_t inputChannels;
};

struct WeightTensorDims3d {
  int64_t outputChannels;
  int64_t flattenedKernelChannels; // kD*kH*kW*C_in
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

struct Spatial3DParam {
  int64_t depth, vertical, horizontal;

  Spatial3DParam(llvm::ArrayRef<int32_t> p)
      : depth(p[0]), vertical(p[1]), horizontal(p[2]) {}
};

struct Conv3dParams {
  Spatial3DParam kernelSize;
  Spatial3DParam stride;
  Spatial3DParam padding;
  int64_t groups;
  llvm::StringRef padding_mode;
};

} // namespace conv3d_verification

namespace conv2d_verification {

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

namespace conv3d_verification {

mlir::LogicalResult verifyTensorRanks(mlir::tt::ttnn::Conv3dOp *op) {
  if (op->getInput().getType().getRank() != 5) {
    return op->emitOpError("input must be a 5D tensor [N, D, H, W, C]");
  }
  if (op->getWeight().getType().getRank() != 2) {
    return op->emitOpError("weight must be a 2D tensor [kD*kH*kW*C, O]");
  }
  if (op->getBias() && op->getBias().getType().getRank() != 2) {
    return op->emitOpError("bias must be a 2D tensor [32, O]");
  }
  if (op->getResult().getType().getRank() != 5) {
    return op->emitOpError("result must be a 5D tensor [N, D_out, H_out, W_out, O]");
  }
  return mlir::success();
}

std::tuple<InputTensorDims3d, WeightTensorDims3d, std::optional<BiasTensorDims3d>>
getConv3dInputDims(mlir::tt::ttnn::Conv3dOp *op) {
  InputTensorDims3d inputDims = {
      op->getBatchSize(), op->getInputDepth(), op->getInputHeight(),
      op->getInputWidth(), op->getInChannels()};

  WeightTensorDims3d weightDims = {
      op->getOutChannels(),
      op->getWeight().getType().getDimSize(WEIGHT_FLATTENED),
      op->getKernelSize()[0], op->getKernelSize()[1], op->getKernelSize()[2]};

  std::optional<BiasTensorDims3d> biasDims;
  if (op->getBias()) {
    biasDims = {op->getBias().getType().getDimSize(BIAS_OUT_CHANNEL)};
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
    return llvm::createStringError(
        "kernel_size must have 3 values, got: " +
        std::to_string(kernelSize.size()));
  }

  if (!llvm::all_of(kernelSize, [](int32_t v) { return v >= 1; })) {
    return llvm::createStringError("kernel_size values must be >= 1, got: " +
                                   formatAttr(kernelSize));
  }

  llvm::ArrayRef<int32_t> stride = op->getStride();
  if (stride.size() != 3) {
    return llvm::createStringError(
        "stride must have 3 values, got: " + std::to_string(stride.size()));
  }
  if (!llvm::all_of(stride, [](int32_t v) { return v >= 1; })) {
    return llvm::createStringError("stride values must be > 0, got: " +
                                   formatAttr(stride));
  }

  llvm::ArrayRef<int32_t> padding = op->getPadding();
  if (padding.size() != 3) {
    return llvm::createStringError(
        "padding must have 3 values, got: " + std::to_string(padding.size()));
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

::mlir::LogicalResult verifyConv3dInputDims(
    mlir::tt::ttnn::Conv3dOp *op, const InputTensorDims3d &inputDims,
    const WeightTensorDims3d &weightDims,
    const std::optional<BiasTensorDims3d> &biasDims,
    const Conv3dParams &params) {

  // Verify bias shape constraints (2D: [32, O])
  if (op->getBias()) {
    auto biasShape = op->getBias().getType().getShape();
    if (biasShape[BIAS_TILE_HEIGHT] != 32) {
      return op->emitOpError("bias first dimension must be 32 (tile height), got ")
             << biasShape[BIAS_TILE_HEIGHT];
    }
  }

  if (inputDims.inputChannels % params.groups != 0) {
    return op->emitOpError() << "in_channels (" << inputDims.inputChannels
                             << ") must be divisible by groups ("
                             << params.groups << ")";
  }

  if (weightDims.outputChannels % params.groups != 0) {
    return op->emitOpError() << "out_channels (" << weightDims.outputChannels
                             << ") must be divisible by groups ("
                             << params.groups << ")";
  }

  int64_t expectedFlattenedDim =
      (inputDims.inputChannels / params.groups) * params.kernelSize.depth *
      params.kernelSize.vertical * params.kernelSize.horizontal;

  if (expectedFlattenedDim != weightDims.flattenedKernelChannels) {
    return op->emitOpError()
           << "weight flattened dimension ("
           << weightDims.flattenedKernelChannels
           << ") must equal kD*kH*kW*C_in/groups (" << expectedFlattenedDim
           << ")";
  }

  if (biasDims && biasDims->outputChannels != weightDims.outputChannels) {
    return op->emitOpError() << "bias output channels ("
                             << biasDims->outputChannels
                             << ") must match weight output channels ("
                             << weightDims.outputChannels << ")";
  }

  return mlir::success();
}

::mlir::LogicalResult verifyConv3dOutputDims(
    mlir::tt::ttnn::Conv3dOp *op, const InputTensorDims3d &inputDims,
    const WeightTensorDims3d &weightDims,
    const std::optional<BiasTensorDims3d> &biasDims,
    const OutputTensorDims3d &outputDims, const Conv3dParams &params) {

  int32_t calculatedDOut =
      (inputDims.inputDepth + 2 * params.padding.depth -
       params.kernelSize.depth) /
          params.stride.depth + 1;

  int32_t calculatedHOut =
      (inputDims.inputHeight + 2 * params.padding.vertical -
       params.kernelSize.vertical) /
          params.stride.vertical + 1;

  int32_t calculatedWOut =
      (inputDims.inputWidth + 2 * params.padding.horizontal -
       params.kernelSize.horizontal) /
          params.stride.horizontal + 1;

  if (outputDims.batchSize != inputDims.batchSize) {
    return op->emitOpError() << "output batch size (" << outputDims.batchSize
                             << ") must match input batch size ("
                             << inputDims.batchSize << ")";
  }

  if (outputDims.outputDepth != calculatedDOut) {
    return op->emitOpError() << "output depth (" << outputDims.outputDepth
                             << ") does not match calculated depth ("
                             << calculatedDOut << ")";
  }

  if (outputDims.outputHeight != calculatedHOut) {
    return op->emitOpError() << "output height (" << outputDims.outputHeight
                             << ") does not match calculated height ("
                             << calculatedHOut << ")";
  }

  if (outputDims.outputWidth != calculatedWOut) {
    return op->emitOpError() << "output width (" << outputDims.outputWidth
                             << ") does not match calculated width ("
                             << calculatedWOut << ")";
  }

  if (outputDims.outputChannels != weightDims.outputChannels) {
    return op->emitOpError()
           << "output channels (" << outputDims.outputChannels
           << ") must match weight output channels ("
           << weightDims.outputChannels << ")";
  }

  return mlir::success();
}

} // namespace conv3d_verification

} // namespace mlir::tt::ttnn::utils::verification_utils

#endif
