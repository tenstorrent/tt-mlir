// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"

::ttnn::Tensor forward() {
  ttnn::MeshDevice* device = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1, 784}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({784, 512}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::ones(::ttnn::Shape({512}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(::ttnn::Shape({256, 10}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::ones(::ttnn::Shape({10}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, *device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::matmul(v1, v2, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::add(v3, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::relu(v9, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::matmul(v10, v4, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::add(v5, v11, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::relu(v12, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::matmul(v13, v6, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::add(v7, v14, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::softmax(v15, -1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  return v16;
}

int32_t main() {
  ttnn::Tensor ret = forward();
  int32_t v1 = 0;
  return v1;
}
