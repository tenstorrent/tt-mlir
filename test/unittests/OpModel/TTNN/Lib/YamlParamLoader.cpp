// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "YamlParamLoader.h"

#include <stdexcept>

#include <vector>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

namespace mlir::tt::ttnn::op_model {
namespace yaml_utils {

mlir::tt::ttnn::TensorMemoryLayout
parseTensorMemoryLayout(const std::string &layout) {
  if (layout == "interleaved") {
    return mlir::tt::ttnn::TensorMemoryLayout::Interleaved;
  }
  if (layout == "height_sharded") {
    return mlir::tt::ttnn::TensorMemoryLayout::HeightSharded;
  }
  if (layout == "width_sharded") {
    return mlir::tt::ttnn::TensorMemoryLayout::WidthSharded;
  }
  if (layout == "block_sharded") {
    return mlir::tt::ttnn::TensorMemoryLayout::BlockSharded;
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
  if (tensorNode["memory_layout"]) {
    testTensor.memoryLayout =
        parseTensorMemoryLayout(tensorNode["memory_layout"].as<std::string>());
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

Conv2dParams parseConv2dParams(const YAML::Node &node) {
  // Parse tensor parameters
  auto inputTensor = parseTestTensor(node["input"]);
  auto weightTensor = parseTestTensor(node["weight"]);
  auto outputTensor = parseTestTensor(node["output"]);

  // Parse scalar parameters
  uint32_t inChannels = node["in_channels"].as<uint32_t>();
  uint32_t outChannels = node["out_channels"].as<uint32_t>();
  uint32_t batchSize = node["batch_size"].as<uint32_t>();
  uint32_t inputHeight = node["input_height"].as<uint32_t>();
  uint32_t inputWidth = node["input_width"].as<uint32_t>();
  uint32_t groups = node["groups"].as<uint32_t>();

  // Parse vector parameters
  auto kernelSize = parseIntVector(node["kernel_size"]);
  auto stride = parseIntVector(node["stride"]);
  auto padding = parseIntVector(node["padding"]);
  auto dilation = parseIntVector(node["dilation"]);

  // Parse expected result
  auto expectedResult = parseExpectedResult(node["expected_result"]);

  return Conv2dParams(inputTensor, weightTensor, outputTensor, inChannels,
                      outChannels, batchSize, inputHeight, inputWidth,
                      kernelSize, stride, padding, dilation, groups,
                      expectedResult);
}

std::vector<Conv2dParams>
parseAllConv2dParams(const std::string &yamlFilePath) {
  YAML::Node config = parseYamlFile(yamlFilePath);
  std::vector<Conv2dParams> params;
  for (const auto &kv : config) {
    params.push_back(parseConv2dParams(kv.second));
  }
  return params;
}

ConvTranspose2dParams parseConvTranspose2dParams(const YAML::Node &node) {
  // Parse tensor parameters
  auto inputTensor = parseTestTensor(node["input"]);
  auto weightTensor = parseTestTensor(node["weight"]);
  auto outputTensor = parseTestTensor(node["output"]);

  // Parse scalar parameters
  uint32_t inChannels = node["in_channels"].as<uint32_t>();
  uint32_t outChannels = node["out_channels"].as<uint32_t>();
  uint32_t batchSize = node["batch_size"].as<uint32_t>();
  uint32_t inputHeight = node["input_height"].as<uint32_t>();
  uint32_t inputWidth = node["input_width"].as<uint32_t>();
  uint32_t groups = node["groups"].as<uint32_t>();

  // Parse vector parameters
  auto kernelSize = parseIntVector(node["kernel_size"]);
  auto stride = parseIntVector(node["stride"]);
  auto padding = parseIntVector(node["padding"]);
  auto outputPadding = parseIntVector(node["output_padding"]);
  auto dilation = parseIntVector(node["dilation"]);

  // Parse expected result
  auto expectedResult = parseExpectedResult(node["expected_result"]);

  return ConvTranspose2dParams(inputTensor, weightTensor, outputTensor,
                               inChannels, outChannels, batchSize, inputHeight,
                               inputWidth, kernelSize, stride, padding,
                               outputPadding, dilation, groups, expectedResult);
}

std::vector<ConvTranspose2dParams>
parseAllConvTranspose2dParams(const std::string &yamlFilePath) {
  YAML::Node config = parseYamlFile(yamlFilePath);
  std::vector<ConvTranspose2dParams> params;
  for (const auto &kv : config) {
    params.push_back(parseConvTranspose2dParams(kv.second));
  }
  return params;
}

PoolingParams parsePoolingParams(const YAML::Node &node) {
  // Parse tensor parameters
  auto inputTensor = parseTestTensor(node["input"]);
  auto outputTensor = parseTestTensor(node["output"]);

  // Parse scalar parameters
  uint32_t batchSize = node["batch_size"].as<uint32_t>();
  uint32_t inputHeight = node["input_height"].as<uint32_t>();
  uint32_t inputWidth = node["input_width"].as<uint32_t>();
  uint32_t inputChannels = node["input_channels"].as<uint32_t>();

  // Parse vector parameters
  auto kernelSize = parseIntVector(node["kernel_size"]);
  auto stride = parseIntVector(node["stride"]);
  auto padding = parseIntVector(node["padding"]);
  auto dilation = parseIntVector(node["dilation"]);
  bool ceilMode = node["ceil_mode"].as<bool>();
  bool inPlaceHalo = node["in_place_halo"].as<bool>();

  // Parse expected result
  auto expectedResult = parseExpectedResult(node["expected_result"]);

  return PoolingParams(inputTensor, outputTensor, batchSize, inputHeight,
                       inputWidth, inputChannels, kernelSize, stride, padding,
                       dilation, ceilMode, inPlaceHalo, expectedResult);
}

std::vector<PoolingParams>
parseAllPoolingParams(const std::string &yamlFilePath) {
  YAML::Node config = parseYamlFile(yamlFilePath);
  std::vector<PoolingParams> params;
  for (const auto &kv : config) {
    params.push_back(parsePoolingParams(kv.second));
  }
  return params;
}

} // namespace yaml_utils

} // namespace mlir::tt::ttnn::op_model
