// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"

::ttnn::Tensor main2(::ttnn::Tensor v1) {
  ttnn::IDevice* v2 = ttnn::DeviceGetter::getInstance();
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v3 = ttnn::mesh_shard(v1);
  ::ttnn::Tensor v4 = ttnn::neg(v3, ::std::nullopt);
  ttnn::deallocate(v3, false);
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v5 = ttnn::mesh_shard(v4);
  ttnn::deallocate(v4, false);
  return v5;
}

::ttnn::Tensor createInputsFor_main2() {
  ttnn::IDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({256, 256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::TILE, ::std::nullopt, ::std::nullopt);
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::std::nullopt);
  return v3;
}

int32_t main() {
  ::ttnn::Tensor v1 = createInputsFor_main2();
  ::ttnn::Tensor v2 = main2(v1);
  int32_t v3 = 0;
  return v3;
}
