// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/tensor.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/work_split.hpp"
#include "core.hpp"
#include "device.hpp"
#include "operations/copy.hpp"
#include "operations/core/core.hpp"
#include "operations/creation.hpp"
#include "operations/eltwise/binary/binary.hpp"
#include "operations/eltwise/binary/binary_composite.hpp"
#include "operations/eltwise/unary/unary_composite.hpp"
#include "operations/embedding/embedding.hpp"
#include "operations/embedding_backward/embedding_backward.hpp"
#include "operations/matmul/matmul.hpp"
#include "operations/normalization/softmax/softmax.hpp"
#include "tensor/tensor.hpp"
#include "tensor/types.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "types.hpp"

#include <cstddef>
#include <iostream>
#include <vector>#include "types.hpp"
#include <iostream>

int32_t main() {
  ttnn::SimpleShape v1s = ttnn::SimpleShape(tt::tt_metal::LegacyShape({1, 32, 50, 100}));
  ttnn::Tensor v1 = ttnn::ones(v1s, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
  ttnn::SimpleShape v2s = ttnn::SimpleShape(tt::tt_metal::LegacyShape({1, 32, 100, 12}));
  ttnn::Tensor v2 = ttnn::ones(v2s, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);

  ttnn::IDevice* v3 = &ttnn::open_device(0);
  ttnn::MemoryConfig v4 = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v5 = ttnn::to_device(v1, v3, v4);
  ttnn::Tensor v6 = ttnn::to_layout(v5, ttnn::Layout::TILE, std::nullopt, std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::MemoryConfig v7 = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v8 = ttnn::to_device(v2, v3, v7);
  ttnn::Tensor v9 = ttnn::to_layout(v8, ttnn::Layout::TILE, std::nullopt, std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::deallocate(v8, false);
  ttnn::SimpleShape v10 = ttnn::SimpleShape(tt::tt_metal::LegacyShape({32, 32, }));
  ttnn::MemoryConfig v11 = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
  ttnn::Tensor v12 = ttnn::empty(v10, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, v3, v11);
  ttnn::Tensor v13 = ttnn::add(v6, v9, std::nullopt, std::nullopt, v12);
  ttnn::deallocate(v9, false);
  ttnn::deallocate(v6, false);
  ttnn::Tensor v14 = ttnn::from_device(v13);
  ttnn::deallocate(v12, false);
  ttnn::Tensor v15 = ttnn::to_layout(v14, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::deallocate(v14, false);
  return v15;
}

std::tuple<ttnn::Tensor, ttnn::Tensor> createInputsFor_add() {
  ttnn::SimpleShape v1 = ttnn::SimpleShape(tt::tt_metal::LegacyShape({32, 32, }));
  ttnn::Tensor v2 = ttnn::ones(v1, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
  ttnn::SimpleShape v3 = ttnn::SimpleShape(tt::tt_metal::LegacyShape({32, 32, }));
  ttnn::Tensor v4 = ttnn::ones(v3, ttnn::DataType::BFLOAT16, ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
  return std::make_tuple(v2, v4);
}

int32_t main() {
  ttnn::Tensor v1;
  ttnn::Tensor v2;
  std::tie(v1, v2) = createInputsFor_add();
  ttnn::Tensor v3 = add(v1, v2);
  int32_t v4 = 0;
  return v4;
}
