// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"

// To generate forward function, run:
// ./build/bin/ttmlir-opt --ttir-to-emitc-pipeline
// test/ttmlir/Silicon/TTNN/emitc/simple_add.mlir | ./build/bin/ttmlir-translate
// --mlir-to-cpp

ttnn::Tensor forward(ttnn::Tensor v1, ttnn::Tensor v2) {
  ttnn::Device *v3 = ttnn::DeviceGetter::getInstance();
  ttnn::MemoryConfig v4 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v5 = ttnn::to_device(v1, v3, v4);
  ttnn::Tensor v6 =
      ttnn::to_layout(v5, ttnn::Layout::TILE, std::nullopt, std::nullopt,
                      static_cast<::ttnn::Device *>(nullptr));
  ttnn::deallocate(v5, false);
  ttnn::MemoryConfig v7 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v8 = ttnn::to_device(v2, v3, v7);
  ttnn::Tensor v9 =
      ttnn::to_layout(v8, ttnn::Layout::TILE, std::nullopt, std::nullopt,
                      static_cast<::ttnn::Device *>(nullptr));
  ttnn::deallocate(v8, false);
  ttnn::Shape v10 = ttnn::Shape(tt::tt_metal::LegacyShape({
      32,
      32,
  }));
  ttnn::MemoryConfig v11 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v12 =
      ttnn::empty(v10, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, v3, v11);
  ttnn::Tensor v13 = ttnn::add(v6, v9, std::nullopt, std::nullopt, v12);
  ttnn::deallocate(v9, false);
  ttnn::deallocate(v6, false);
  ttnn::Tensor v14 = ttnn::from_device(v13);
  ttnn::deallocate(v12, false);
  ttnn::Tensor v15 =
      ttnn::to_layout(v14, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt,
                      static_cast<::ttnn::Device *>(nullptr));
  ttnn::deallocate(v14, false);
  return v15;
}

int main() {
  // Create shapes
  //
  const size_t tensor_height = 32;
  const size_t tensor_width = 32;
  ttnn::Shape xs =
      ttnn::Shape(tt::tt_metal::LegacyShape{1, 1, tensor_height, tensor_width});
  ttnn::Shape ys =
      ttnn::Shape(tt::tt_metal::LegacyShape{1, 1, tensor_height, tensor_width});

  // Create tensors on cpu
  //
  auto x = ttnn::ones(xs, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
  auto y = ttnn::ones(ys, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

  // Run fwd pass on device
  //
  ttnn::Tensor result = forward(x, y);

  // Print result
  //
  result.print();
}
