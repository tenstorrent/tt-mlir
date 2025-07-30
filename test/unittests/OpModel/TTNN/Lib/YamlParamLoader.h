// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

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

using Conv2dParams = std::tuple<detail::TestTensor,         // input
                                detail::TestTensor,         // weight
                                detail::TestTensor,         // output
                                uint32_t,                   // in_channels
                                uint32_t,                   // out_channels
                                uint32_t,                   // batch_size
                                uint32_t,                   // input_height
                                uint32_t,                   // input_width
                                llvm::SmallVector<int32_t>, // kernel_size
                                llvm::SmallVector<int32_t>, // stride
                                llvm::SmallVector<int32_t>, // padding
                                llvm::SmallVector<int32_t>, // dilation
                                uint32_t,                   // groups
                                detail::ExpectedResult>;

namespace yaml_utils {

/// Parse tensor layout string to enum
mlir::tt::ttnn::TensorMemoryLayout parseTensorLayout(const std::string &layout);

/// Parse buffer type string to enum
mlir::tt::ttnn::BufferType parseBufferType(const std::string &bufferType);

/// Parse Conv2d parameters from YAML file
YAML::Node parseYamlFile(const std::string &yamlFilePath);

/// Returns tuple matching OpModelConv2dParam test parameter structure
Conv2dParams parseConv2dParams(const YAML::Node &conv2dParams);

} // namespace yaml_utils

} // namespace mlir::tt::op_model::ttnn
