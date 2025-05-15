// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"

#include <chrono>
#include <iostream>

::ttnn::Tensor add(::ttnn::Tensor v1, ::ttnn::Tensor v2) {
  ::ttnn::Tensor v3 = ttnn::add(v1, v2, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v2, false);
  ttnn::deallocate(v1, false);
  return v3;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for_add() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({32, 32}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({32, 32}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  return std::make_tuple(v3, v5);
}

int32_t main() {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  std::tie(v1, v2) = create_inputs_for_add();

  auto start = std::chrono::high_resolution_clock::now();
  ::ttnn::Tensor v3 = add(v1, v2);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration_in_seconds = end - start;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "add(...) took " << duration_in_seconds.count() << " seconds" << std::endl;
  std::cout << "add(...) took " << duration << " microseconds" << std::endl;

  int32_t v4 = 0;
  return v4;
}
