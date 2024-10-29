// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"

// To generate forward function, run:
// ./build/bin/ttmlir-opt --ttir-load-system-desc --ttir-implicit-device
// --ttir-layout --convert-ttir-to-ttnn --convert-ttnn-to-emitc
// test/ttmlir/Dialect/TTNN/simple_multiply.mlir | ./build/bin/ttmlir-translate
// -mlir-to-cpp -allow-unregistered-dialect

// Forward function example
//
ttnn::Tensor forward(ttnn::Tensor v1, ttnn::Tensor v2) {
  //   ttnn::Tensor v1 = inputs[0];
  //   ttnn::Tensor v2 = inputs[1];
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
      32,
      32,
  }));
  ttnn::MemoryConfig v12 = ttnn::MemoryConfig(
      ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v13 =
      ttnn::empty(v10, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, v3, v12);
  //   std::cout << "input0:" << std::endl;
  //   v6.print();
  //   std::cout << "input1:" << std::endl;
  //   v9.print();
  //   std::cout << std::endl << std::endl;
  ttnn::Tensor v14 = ttnn::add(v6, v9, std::nullopt, std::nullopt, v13);
  //   v13.print();
  ttnn::Tensor v15 = ttnn::to_layout(v14, ttnn::Layout::ROW_MAJOR, std::nullopt,
                                     std::nullopt, v3);
  ttnn::Tensor v16 = ttnn::from_device(v15);
  //   std::cout << "Output:" << std::endl;
  //   v16.print();
  //   std::cout << std::endl << std::endl;
  return v16;
  //   return std::vector<ttnn::Tensor>{v16};
}

int main() {
  // Create shapes
  //
  const size_t tensor_height = 32;
  const size_t tensor_width = 32;
  ttnn::Shape xs =
      ttnn::Shape(tt::tt_metal::LegacyShape{tensor_height, tensor_width});
  ttnn::Shape ys =
      ttnn::Shape(tt::tt_metal::LegacyShape{tensor_height, tensor_width});

  // Create tensors on cpu
  //
  //   auto x = ttnn::full(xs, 4.0f, ttnn::DataType::BFLOAT16,
  //   ttnn::Layout::TILE); auto y = ttnn::full(ys, 6.0f,
  //   ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
  auto x = ttnn::ones(xs, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
  auto y = ttnn::ones(ys, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

  // Run fwd pass on device
  //
  ttnn::Tensor result = forward(x, y);

  // Print result
  //
  result.print();
}
