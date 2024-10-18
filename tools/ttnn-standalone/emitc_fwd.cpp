// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "emitc_fwd.h"

#include "ttnn-precompiled.hpp"

// To generate forward function, run:
// ./build/bin/ttmlir-opt --ttir-load-system-desc --ttir-implicit-device
// --ttir-layout --convert-ttir-to-ttnn --convert-ttnn-to-emitc
// test/ttmlir/Dialect/TTNN/simple_multiply.mlir | ./build/bin/ttmlir-translate
// -mlir-to-cpp -allow-unregistered-dialect

// Forward function example
//
std::vector<ttnn::Tensor> forward(std::vector<ttnn::Tensor> inputs) {
  ttnn::Tensor v1 = inputs[0];
  ttnn::Tensor v2 = inputs[1];
  ttnn::Device *v3 = ttnn::DeviceGetter::getInstance();
  ttnn::Tensor v4 =
      ttnn::to_layout(v1, ttnn::Layout::TILE, std::nullopt, std::nullopt, v3);
  ttnn::MemoryConfig v5 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v6 = ttnn::to_device(v4, v3, v5);
  ttnn::Tensor v7 =
      ttnn::to_layout(v2, ttnn::Layout::TILE, std::nullopt, std::nullopt, v3);
  ttnn::MemoryConfig v8 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v9 = ttnn::to_device(v7, v3, v8);
  ttnn::Shape v10 = ttnn::Shape(tt::tt_metal::LegacyShape({
      64,
      128,
  }));
  ttnn::MemoryConfig v12 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v13 = ttnn::empty(v10, ttnn::DataType::FLOAT32,
                                 ttnn::Layout::ROW_MAJOR, v3, v12);
  ttnn::Tensor v14 = ttnn::multiply(v6, v9, std::nullopt, std::nullopt, v13);
  ttnn::Tensor v15 = ttnn::to_layout(v14, ttnn::Layout::ROW_MAJOR, std::nullopt,
                                     std::nullopt, v3);
  ttnn::Tensor v16 = ttnn::from_device(v15);
  return std::vector<ttnn::Tensor>{v16};
}
