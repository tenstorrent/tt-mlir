// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

template <typename... T>
std::vector<ttnn::Tensor> utilCreateVec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

#include "ttnn-precompiled.hpp"
std::vector<ttnn::Tensor> add(std::vector<ttnn::Tensor> v1, ttnn::IDevice* v2) {
  ttnn::Tensor v3 = v1[0];
  ttnn::Tensor v4 = v1[1];
  ttnn::MemoryConfig v5 = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v6 = ttnn::to_device(v3, v2, v5);
  ttnn::Tensor v7 = ttnn::to_layout(v6, ttnn::Layout::TILE, std::nullopt, std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::deallocate(v6, false);
  ttnn::MemoryConfig v8 = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v9 = ttnn::to_device(v4, v2, v8);
  ttnn::Tensor v10 = ttnn::to_layout(v9, ttnn::Layout::TILE, std::nullopt, std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::deallocate(v9, false);
  ttnn::Shape v11 = ttnn::Shape(tt::tt_metal::LegacyShape({32, 32, }));
  ttnn::MemoryConfig v12 = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v13 = ttnn::empty(v11, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, v2, v12);
  ttnn::Tensor v14 = ttnn::add(v7, v10, std::nullopt, std::nullopt, v13);
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v7, false);
  ttnn::Tensor v15 = ttnn::from_device(v14);
  ttnn::deallocate(v13, false);
  ttnn::Tensor v16 = ttnn::to_layout(v15, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::deallocate(v15, false);
  std::vector<ttnn::Tensor> v17 = utilCreateVec(v16);
  return v17;
}
