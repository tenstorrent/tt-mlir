// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"
ttnn::Tensor add(ttnn::Tensor v1, ttnn::Tensor v2) {
  ttnn::Tensor v3 = ttnn::add(v1, v2, std::nullopt, std::nullopt);
  return v3;
}

std::tuple<ttnn::Tensor, ttnn::Tensor> createInputsFor_add() {
  ttnn::IDevice* v1 = ttnn::DeviceGetter::getInstance();
  ttnn::Shape v2 = ttnn::Shape({32, 32});
  ttnn::Tensor v3 = ttnn::ones(v2, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, std::nullopt, std::nullopt);
  ttnn::Tensor v4 = ttnn::to_device(v3, v1, std::nullopt);
  ttnn::Shape v5 = ttnn::Shape({32, 32});
  ttnn::Tensor v6 = ttnn::ones(v5, ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, std::nullopt, std::nullopt);
  ttnn::Tensor v7 = ttnn::to_device(v6, v1, std::nullopt);
  return std::make_tuple(v4, v7);
}

int32_t main() {
  ttnn::Tensor v1;
  ttnn::Tensor v2;
  std::tie(v1, v2) = createInputsFor_add();
  ttnn::Tensor v3 = add(v1, v2);
  int32_t v4 = 0;
  return v4;
}
