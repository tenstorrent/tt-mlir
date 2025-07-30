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

namespace mlir::tt::op_model::ttnn {
namespace detail {

struct TestTensor {
  llvm::SmallVector<int64_t> shape;
  mlir::tt::ttnn::TensorMemoryLayout layout;
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

namespace yaml_utils {

/// Parse tensor layout string to enum
mlir::tt::ttnn::TensorMemoryLayout parseTensorLayout(const std::string &layout);

/// Parse buffer type string to enum
mlir::tt::ttnn::BufferType parseBufferType(const std::string &bufferType);

/// Parse Conv2d parameters from YAML file
YAML::Node parseYamlFile(const std::string &yamlFilePath);

/// Returns Conv2dParams object matching OpModelConv2dParam test parameter
/// structure
Conv2dParams parseConv2dParams(const YAML::Node &conv2dParams);

} // namespace yaml_utils

} // namespace mlir::tt::op_model::ttnn
