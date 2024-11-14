// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-dylib.hpp"

// Forward function example
//
std::vector<ttnn::Tensor> forward(std::vector<ttnn::Tensor> inputs) {
  ttnn::Tensor v1 = inputs[0];
  ttnn::Tensor v2 = inputs[1];
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
  return std::vector<ttnn::Tensor>{v15};
}
