// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pch.hpp"

// ttnn::Tensor forward(ttnn::Tensor v1, ttnn::Tensor v2) {
//   ttnn::Device v3 = ttnn::open_device(0);
//   ttnn::Tensor v4 = ttnn::full(v3);
//   ttnn::Tensor v5 = ttnn::to_memory_config(v1, v4);
//   ttnn::Tensor v6 = ttnn::full(v3);
//   ttnn::Tensor v7 = ttnn::to_memory_config(v2, v6);
//   ttnn::Tensor v8 = ttnn::full(v3);
//   ttnn::Tensor v9 = ttnn::multiply(v5, v7, v8);
//   ttnn::Tensor v10 = ttnn::full(v3);
//   ttnn::Tensor v11 = ttnn::to_memory_config(v9, v10);
//   ttnn::close_device(v3);
//   return v11;
// }

ttnn::Tensor forward(ttnn::Tensor v1, ttnn::Tensor v2) {
  ttnn::device::Device *v3 = &ttnn::device::open_device(0);

  ttnn::Tensor v4 = ttnn::full(v1.shape(), 4.0f, v1.tensor_attributes->dtype,
                               v1.tensor_attributes->layout, std::ref(*v3));
  //   ttnn::Tensor v5 = ttnn::to_memory_config(v1, v4);
  ttnn::Tensor v6 = ttnn::full(v2.shape(), 6.0f, v2.tensor_attributes->dtype,
                               v2.tensor_attributes->layout, std::ref(*v3));
  //   ttnn::Tensor v7 = ttnn::to_memory_config(v2, v6);

  //   ttnn::Tensor v8 = ttnn::full(v3);
  ttnn::Tensor v9 = ttnn::multiply(v4, v6);
  v9 = v9.cpu(); // move to CPU

  v3->close();
  return v9;
}

int main() {
  // Create shapes
  //
  const size_t tensor_width = 32;
  const size_t tensor_height = 32;
  ttnn::Shape xs = ttnn::Shape(Shape{1, 1, tensor_width, tensor_height});
  ttnn::Shape ys = ttnn::Shape(Shape{1, 1, tensor_width, tensor_height});

  // Create tensors on cpu
  //
  auto x = ttnn::full(xs, 4.0f, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);
  auto y = ttnn::full(ys, 6.0f, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE);

  // Run fwd pass on device
  //
  ttnn::Tensor result = forward(x, y);

  // Print result
  //
  result.print();
}
