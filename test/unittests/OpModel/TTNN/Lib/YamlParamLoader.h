// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

#include <yaml-cpp/yaml.h>

namespace mlir::tt::ttnn::op_model {
namespace detail {

struct TestTensor {
  llvm::SmallVector<int64_t> shape;
  mlir::tt::ttnn::TensorMemoryLayout memoryLayout;
  mlir::tt::ttnn::BufferType bufferType;
  std::optional<llvm::SmallVector<int64_t>> virtualGrid = std::nullopt;
};

struct ExpectedResult {
  bool expectedLegal = false;
  size_t expectedCbSize = 0;
  size_t expectedPeakSize = 0;
  size_t expectedOutputSize = 0;
};

} // namespace detail

struct Conv2dParams {
  detail::TestTensor input;
  detail::TestTensor weight;
  detail::TestTensor output;
  uint32_t inChannels;
  uint32_t outChannels;
  uint32_t batchSize;
  uint32_t inputHeight;
  uint32_t inputWidth;
  llvm::SmallVector<int32_t> kernelSize;
  llvm::SmallVector<int32_t> stride;
  llvm::SmallVector<int32_t> padding;
  llvm::SmallVector<int32_t> dilation;
  uint32_t groups;
  detail::ExpectedResult expectedResult;

  Conv2dParams(detail::TestTensor input, detail::TestTensor weight,
               detail::TestTensor output, uint32_t inChannels,
               uint32_t outChannels, uint32_t batchSize, uint32_t inputHeight,
               uint32_t inputWidth, llvm::SmallVector<int32_t> kernelSize,
               llvm::SmallVector<int32_t> stride,
               llvm::SmallVector<int32_t> padding,
               llvm::SmallVector<int32_t> dilation, uint32_t groups,
               detail::ExpectedResult expectedResult)
      : input(std::move(input)), weight(std::move(weight)),
        output(std::move(output)), inChannels(inChannels),
        outChannels(outChannels), batchSize(batchSize),
        inputHeight(inputHeight), inputWidth(inputWidth),
        kernelSize(std::move(kernelSize)), stride(std::move(stride)),
        padding(std::move(padding)), dilation(std::move(dilation)),
        groups(groups), expectedResult(std::move(expectedResult)) {}
};

struct ConvTranspose2dParams {
  detail::TestTensor input;
  detail::TestTensor weight;
  detail::TestTensor output;
  uint32_t inChannels;
  uint32_t outChannels;
  uint32_t batchSize;
  uint32_t inputHeight;
  uint32_t inputWidth;
  llvm::SmallVector<int32_t> kernelSize;
  llvm::SmallVector<int32_t> stride;
  llvm::SmallVector<int32_t> padding;
  llvm::SmallVector<int32_t> outputPadding;
  llvm::SmallVector<int32_t> dilation;
  uint32_t groups;
  detail::ExpectedResult expectedResult;

  ConvTranspose2dParams(detail::TestTensor input, detail::TestTensor weight,
                        detail::TestTensor output, uint32_t inChannels,
                        uint32_t outChannels, uint32_t batchSize,
                        uint32_t inputHeight, uint32_t inputWidth,
                        llvm::SmallVector<int32_t> kernelSize,
                        llvm::SmallVector<int32_t> stride,
                        llvm::SmallVector<int32_t> padding,
                        llvm::SmallVector<int32_t> outputPadding,
                        llvm::SmallVector<int32_t> dilation, uint32_t groups,
                        detail::ExpectedResult expectedResult)
      : input(std::move(input)), weight(std::move(weight)),
        output(std::move(output)), inChannels(inChannels),
        outChannels(outChannels), batchSize(batchSize),
        inputHeight(inputHeight), inputWidth(inputWidth),
        kernelSize(std::move(kernelSize)), stride(std::move(stride)),
        padding(std::move(padding)), outputPadding(std::move(outputPadding)),
        dilation(std::move(dilation)), groups(groups),
        expectedResult(std::move(expectedResult)) {}
};

struct PoolingParams {
  detail::TestTensor input;
  detail::TestTensor output;
  uint32_t batchSize;
  uint32_t inputHeight;
  uint32_t inputWidth;
  uint32_t inputChannels;
  llvm::SmallVector<int32_t> kernelSize;
  llvm::SmallVector<int32_t> stride;
  llvm::SmallVector<int32_t> padding;
  llvm::SmallVector<int32_t> dilation;
  bool ceilMode;
  bool inPlaceHalo;
  detail::ExpectedResult expectedResult;

  PoolingParams(detail::TestTensor input, detail::TestTensor output,
                uint32_t batchSize, uint32_t inputHeight, uint32_t inputWidth,
                uint32_t inputChannels, llvm::SmallVector<int32_t> kernelSize,
                llvm::SmallVector<int32_t> stride,
                llvm::SmallVector<int32_t> padding,
                llvm::SmallVector<int32_t> dilation, bool ceilMode,
                bool inPlaceHalo, detail::ExpectedResult expectedResult)
      : input(std::move(input)), output(std::move(output)),
        batchSize(batchSize), inputHeight(inputHeight), inputWidth(inputWidth),
        inputChannels(inputChannels), kernelSize(std::move(kernelSize)),
        stride(std::move(stride)), padding(std::move(padding)),
        dilation(std::move(dilation)), ceilMode(ceilMode),
        inPlaceHalo(inPlaceHalo), expectedResult(std::move(expectedResult)) {}
};

namespace yaml_utils {

/// Parse tensor layout string to enum
mlir::tt::ttnn::TensorMemoryLayout
parseTensorMemoryLayout(const std::string &layout);

/// Parse buffer type string to enum
mlir::tt::ttnn::BufferType parseBufferType(const std::string &bufferType);

/// Parse Conv2d parameters from YAML file
YAML::Node parseYamlFile(const std::string &yamlFilePath);

/// Returns Conv2dParams object matching OpModelConv2dParam test parameter
/// structure
Conv2dParams parseConv2dParams(const YAML::Node &node);

/// Returns a vector of Conv2dParams objects from a YAML file
std::vector<Conv2dParams> parseAllConv2dParams(const std::string &yamlFilePath);

/// Returns ConvTranspose2dParams object matching OpModelConvTranspose2dParam
/// test parameter structure
ConvTranspose2dParams parseConvTranspose2dParams(const YAML::Node &node);

/// Returns a vector of ConvTranspose2dParams objects from a YAML file
std::vector<ConvTranspose2dParams>
parseAllConvTranspose2dParams(const std::string &yamlFilePath);

/// Returns MaxPool2dParams object matching OpModelMaxPool2dParam test
/// parameter structure
PoolingParams parsePoolingParams(const YAML::Node &node);

/// Returns a vector of MaxPool2dParams objects from a YAML file
std::vector<PoolingParams>
parseAllPoolingParams(const std::string &yamlFilePath);

} // namespace yaml_utils

} // namespace mlir::tt::ttnn::op_model
