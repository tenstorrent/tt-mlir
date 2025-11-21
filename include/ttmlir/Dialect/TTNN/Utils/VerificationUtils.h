// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_VERIFICATIONUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_VERIFICATIONUTILS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn::utils::verification_utils {
namespace conv2d {
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

mlir::LogicalResult verifyTensorRanks(mlir::tt::ttnn::Conv2dOp *op);

std::tuple<InputTensorDims, WeightTensorDims, std::optional<BiasTensorDims>>
getConv2dInputDims(mlir::tt::ttnn::Conv2dOp *op);

OutputTensorDims getConv2dOutputDims(mlir::tt::ttnn::Conv2dOp *op);

llvm::Expected<Conv2dParams>
getAndVerifyConv2dParams(mlir::tt::ttnn::Conv2dOp *op);

::mlir::LogicalResult verifyConv2dInputDims(
    mlir::tt::ttnn::Conv2dOp *op, const InputTensorDims &inputDims,
    const WeightTensorDims &weightDims,
    const std::optional<BiasTensorDims> &biasDims, const Conv2dParams &params);

::mlir::LogicalResult verifyOutputDimensions(
    mlir::tt::ttnn::Conv2dOp *op, const InputTensorDims &inputDims,
    const WeightTensorDims &weightDims,
    const std::optional<BiasTensorDims> &biasDims,
    const OutputTensorDims &outputDims, const Conv2dParams &params);

} // namespace conv2d

namespace conv3d {
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
  WEIGHT_FLATTENED = 0,  // kD*kH*kW*C_in/groups (patch_size)
  WEIGHT_OUT_CHANNEL = 1 // O (output channels)
};

// Bias is 2D: [1, O]
enum BiasDim : unsigned {
  BIAS_FIRST_DIM = 0,  // Must be 1
  BIAS_OUT_CHANNEL = 1 // Must be O (output channels)
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
  int64_t flattenedKernelChannels; // kD*kH*kW*C_in/groups
  int64_t kernelDepth;
  int64_t kernelHeight;
  int64_t kernelWidth;
};

struct BiasTensorDims3d {
  int64_t firstDim;
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

mlir::LogicalResult verifyTensorRanks(mlir::tt::ttnn::Conv3dOp *op);

std::tuple<InputTensorDims3d, WeightTensorDims3d,
           std::optional<BiasTensorDims3d>>
getConv3dInputDims(mlir::tt::ttnn::Conv3dOp *op);

OutputTensorDims3d getConv3dOutputDims(mlir::tt::ttnn::Conv3dOp *op);

llvm::Expected<Conv3dParams>
getAndVerifyConv3dParams(mlir::tt::ttnn::Conv3dOp *op);

::mlir::LogicalResult
verifyConv3dInputDims(mlir::tt::ttnn::Conv3dOp *op,
                      const InputTensorDims3d &inputDims,
                      const WeightTensorDims3d &weightDims,
                      const std::optional<BiasTensorDims3d> &biasDims,
                      const Conv3dParams &params);

::mlir::LogicalResult verifyConv3dOutputDims(
    mlir::tt::ttnn::Conv3dOp *op, const InputTensorDims3d &inputDims,
    const WeightTensorDims3d &weightDims, const OutputTensorDims3d &outputDims,
    const Conv3dParams &params);

} // namespace conv3d

} // namespace mlir::tt::ttnn::utils::verification_utils

#endif
