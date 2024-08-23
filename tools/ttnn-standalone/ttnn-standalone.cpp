// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/buffer.hpp"
#include "operations/core/core.hpp"
#include "tensor/types.hpp"
#include "ttnn-precompiled.hpp"
#include "types.hpp"

// Below is a snippet generated with:
// ./build/bin/ttmlir-opt --ttir-load-system-desc --ttir-layout
// --ttnn-open-device --convert-ttir-to-ttnn --convert-ttnn-to-emitc
// test/ttmlir/Dialect/TTNN/simple_multiply.mlir | ./build/bin/ttmlir-translate
// -mlir-to-cpp -allow-unregistered-dialect
//
// #include "pch.hpp"
// ttnn::Tensor forward(ttnn::Tensor v1, ttnn::Tensor v2) {
//   ttnn::device::Device& v3 = ttnn::device::open_device(0);
//   ttnn::Tensor v4 = ttnn::full(v3);
//   ttnn::Tensor v5 = ttnn::to_memory_config(v1, v4);
//   ttnn::Tensor v6 = ttnn::full(v3);
//   ttnn::Tensor v7 = ttnn::to_memory_config(v2, v6);
//   ttnn::Tensor v8 = ttnn::full(v3);
//   ttnn::Tensor v9 = ttnn::multiply(v5, v7, v8);
//   ttnn::Tensor v10 = ttnn::full(v3);
//   ttnn::Tensor v11 = ttnn::to_memory_config(v9, v10);
//   ttnn::device::close_device(v3);
//   return v11;
// }

ttnn::Tensor forward(ttnn::Tensor v1, ttnn::Tensor v2) {
  ttnn::Device &v3 = ttnn::open_device(0);

  // ttnn::Tensor v4 = ttnn::empty(v1.shape(), v1.tensor_attributes->dtype,
  //                               v1.tensor_attributes->layout, v3);

  MemoryConfig memConfig = ttnn::MemoryConfig{
      .memory_layout = ttnn::TensorMemoryLayout::SINGLE_BANK,
      .buffer_type = ttnn::BufferType::DRAM,
      // .shard_spec = std::nullopt,
  };

  ttnn::Tensor v4 = ttnn::to_layout(v1, ttnn::Layout::ROW_MAJOR,
                                    ttnn::DataType::BFLOAT16, memConfig, &v3);
  ttnn::Tensor v5 = ttnn::to_layout(v2, ttnn::Layout::ROW_MAJOR,
                                    ttnn::DataType::BFLOAT16, memConfig, &v3);

  //   ttnn::Tensor v5 = ttnn::to_memory_config(v1, v4);
  //   struct MemoryConfig {
  //     TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;  //
  //     Interleave the data across multiple banks BufferType buffer_type =
  //     BufferType::DRAM;                           // Can be either DRAM or L1
  //     std::optional<ShardSpec> shard_spec = std::nullopt;
  //     bool is_sharded() const;
  //     bool is_l1() const;
  //     bool is_dram() const;
  // };
  // auto q = MemoryConfig();
  // ttnn::to_layout
  // q.memory_layout = TensorMemoryLayout::INTERLEAVED;

  // ttnn::to_memory_config(v4, q, std::optional<DataType>());

  // ttnn::Tensor v6 = ttnn::empty(v2.shape(), v2.tensor_attributes->dtype,
  //                               v2.tensor_attributes->layout, v3);
  //   ttnn::Tensor v7 = ttnn::to_memory_config(v2, v6);

  ttnn::Tensor v9 = ttnn::empty(v2.shape(), v2.tensor_attributes->dtype,
                                v2.tensor_attributes->layout, v3);

  ttnn::Tensor v10 = ttnn::to_layout(v9, ttnn::Layout::TILE,
                                     ttnn::DataType::BFLOAT16, memConfig, &v3);

  ttnn::multiply(v4, v5, std::nullopt, std::nullopt, v10);
  v10 = v10.cpu(); // move to CPU

  ttnn::close_device(v3);
  return v10;
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
