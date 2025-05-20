// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"
::ttnn::Tensor _main(::ttnn::Tensor v1, ::ttnn::Tensor v2, ::ttnn::Tensor v3, ::ttnn::Tensor v4, ::ttnn::Tensor v5) {
  ttnn::distributed::MeshDevice* v6 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v7 = ttnn::matmul(v1, v5, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v8 = ttnn::add(v7, v4, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::tt::tt_metal::ShardSpec(::ttnn::CoreRangeSet {std::set<CoreRange> {::ttnn::CoreRange {::ttnn::CoreCoord {0, 0}, ::ttnn::CoreCoord {7, 0}}, }}, {32, 32}, ::ttnn::ShardOrientation::ROW_MAJOR)});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v9 = ttnn::relu(v8, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::tt::tt_metal::ShardSpec(::ttnn::CoreRangeSet {std::set<CoreRange> {::ttnn::CoreRange {::ttnn::CoreCoord {0, 0}, ::ttnn::CoreCoord {7, 0}}, }}, {32, 32}, ::ttnn::ShardOrientation::ROW_MAJOR)});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::to_memory_config(v9, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::matmul(v10, v3, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v3, false);
  ::ttnn::Tensor v12 = ttnn::add(v11, v2, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::tt::tt_metal::ShardSpec(::ttnn::CoreRangeSet {std::set<CoreRange> {::ttnn::CoreRange {::ttnn::CoreCoord {0, 0}, ::ttnn::CoreCoord {0, 0}}, }}, {32, 32}, ::ttnn::ShardOrientation::ROW_MAJOR)});
  ttnn::deallocate(v11, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v13 = ttnn::to_memory_config(v12, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::softmax(v13, 1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v13, false);
  return v14;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> create_inputs_for__main() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 784}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1, 10}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(::ttnn::Shape({256, 10}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_device(v6, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(::ttnn::Shape({1, 256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::to_device(v8, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(::ttnn::Shape({784, 256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::to_device(v10, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  return std::make_tuple(v3, v5, v7, v9, v11);
}

int32_t main() {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  ::ttnn::Tensor v3;
  ::ttnn::Tensor v4;
  ::ttnn::Tensor v5;
  std::tie(v1, v2, v3, v4, v5) = create_inputs_for__main();
  ::ttnn::Tensor v6 = _main(v1, v2, v3, v4, v5);
  int32_t v7 = 0;
  return v7;
}
