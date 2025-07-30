// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "YamlParamLoader.h"

#include <stdexcept>

#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

namespace mlir::tt::op_model::ttnn {
namespace yaml_utils {

mlir::tt::ttnn::TensorMemoryLayout
parseTensorLayout(const std::string &layout) {
  if (layout == "row_major") {
    return mlir::tt::ttnn::TensorMemoryLayout::Interleaved;
  }
  if (layout == "tile") {
    return mlir::tt::ttnn::TensorMemoryLayout::Interleaved;
  }
  throw std::runtime_error("Unknown tensor layout: " + layout);
}

mlir::tt::ttnn::BufferType parseBufferType(const std::string &bufferType) {
  if (bufferType == "dram") {
    return mlir::tt::ttnn::BufferType::DRAM;
  }
  if (bufferType == "system_memory") {
    return mlir::tt::ttnn::BufferType::SystemMemory;
  }
  if (bufferType == "l1") {
    return mlir::tt::ttnn::BufferType::L1;
  }
  throw std::runtime_error("Unknown buffer type: " + bufferType);
}

detail::TestTensor parseTestTensor(const YAML::Node &tensorNode) {
  detail::TestTensor testTensor;

  // Parse shape
  if (tensorNode["shape"]) {
    for (const auto &dim : tensorNode["shape"]) {
      testTensor.shape.push_back(dim.as<int64_t>());
    }
  }

  // Parse tensor layout
  if (tensorNode["tensor_layout"]) {
    testTensor.layout =
        parseTensorLayout(tensorNode["tensor_layout"].as<std::string>());
  }

  // Parse buffer type
  if (tensorNode["buffer_type"]) {
    testTensor.bufferType =
        parseBufferType(tensorNode["buffer_type"].as<std::string>());
  }

  // Parse virtual grid (optional)
  if (tensorNode["virtual_grid"]) {
    testTensor.virtualGrid = llvm::SmallVector<int64_t>();
    for (const auto &dim : tensorNode["virtual_grid"]) {
      testTensor.virtualGrid->push_back(dim.as<int64_t>());
    }
  }

  return testTensor;
}

llvm::SmallVector<int32_t> parseIntVector(const YAML::Node &node) {
  llvm::SmallVector<int32_t> result;
  for (const auto &elem : node) {
    result.push_back(elem.as<int32_t>());
  }
  return result;
}

detail::ExpectedResult parseExpectedResult(const YAML::Node &resultNode) {
  detail::ExpectedResult result;

  if (resultNode["legal"]) {
    result.expectedLegal = resultNode["legal"].as<bool>();
  }
  if (resultNode["cb_size"]) {
    result.expectedCbSize = resultNode["cb_size"].as<size_t>();
  }
  if (resultNode["peak_size"]) {
    result.expectedPeakSize = resultNode["peak_size"].as<size_t>();
  }
  if (resultNode["output_size"]) {
    result.expectedOutputSize = resultNode["output_size"].as<size_t>();
  }

  return result;
}

YAML::Node parseYamlFile(const std::string &yamlFilePath) {
  YAML::Node config = YAML::LoadFile(yamlFilePath);
  return config;
}

Conv2dParams parseConv2dParams(const YAML::Node &conv2dParams) {
  // Parse tensor parameters
  auto inputTensor = parseTestTensor(conv2dParams["input"]);
  auto weightTensor = parseTestTensor(conv2dParams["weight"]);
  auto outputTensor = parseTestTensor(conv2dParams["output"]);

  // Parse scalar parameters
  uint32_t inChannels = conv2dParams["in_channels"].as<uint32_t>();
  uint32_t outChannels = conv2dParams["out_channels"].as<uint32_t>();
  uint32_t batchSize = conv2dParams["batch_size"].as<uint32_t>();
  uint32_t inputHeight = conv2dParams["input_height"].as<uint32_t>();
  uint32_t inputWidth = conv2dParams["input_width"].as<uint32_t>();
  uint32_t groups = conv2dParams["groups"].as<uint32_t>();

  // Parse vector parameters
  auto kernelSize = parseIntVector(conv2dParams["kernel_size"]);
  auto stride = parseIntVector(conv2dParams["stride"]);
  auto padding = parseIntVector(conv2dParams["padding"]);
  auto dilation = parseIntVector(conv2dParams["dilation"]);

  // Parse expected result
  auto expectedResult = parseExpectedResult(conv2dParams["expected_result"]);

  return std::make_tuple(inputTensor, weightTensor, outputTensor, inChannels,
                         outChannels, batchSize, inputHeight, inputWidth,
                         kernelSize, stride, padding, dilation, groups,
                         expectedResult);
}

} // namespace yaml_utils

} // namespace mlir::tt::op_model::ttnn
