#include <iostream>
#include <chrono>
#include "ttnn-precompiled.hpp"
::std::vector<::ttnn::Tensor>
forward_const_eval_0(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_1(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_2(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_3(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_4(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_5(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_6(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_7(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_8(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_9(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_10(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_11(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_12(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::std::vector<::ttnn::Tensor> v3 = util_create_vec(v2);
  return v3;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_13(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_14(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_15(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_16(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_17(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_18(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_19(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_20(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_21(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_22(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_23(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::std::vector<::ttnn::Tensor> v3 = util_create_vec(v2);
  return v3;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_24(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_25(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_26(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_27(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_28(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_29(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_30(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_31(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_32(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_33(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_34(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_35(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_36(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::std::vector<::ttnn::Tensor> v3 = util_create_vec(v2);
  return v3;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_37(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_38(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_39(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_40(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_41(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_42(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::std::vector<::ttnn::Tensor> v3 = util_create_vec(v2);
  return v3;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_43(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_44(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_45(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_46(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_47(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_48(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_49(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_50(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_51(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_52(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_53(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_54(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_55(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_56(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
forward_const_eval_57(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::permute(
      v5, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_0;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_1;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_2;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_3;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_4;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_5;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_6;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_7;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_8;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_9;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_10;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_11;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_12;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_13;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_14;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_15;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_16;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_17;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_18;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_19;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_20;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_21;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_22;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_23;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_24;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_25;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_26;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_27;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_28;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_29;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_30;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_31;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_32;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_33;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_34;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_35;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_36;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_37;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_38;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_39;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_40;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_41;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_42;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_43;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_44;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_45;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_46;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_47;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_48;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_49;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_50;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_51;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_52;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_53;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_54;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_55;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_56;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_57;
::std::vector<::ttnn::Tensor> forward(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ::ttnn::Tensor v4 = v1[2];
  ::ttnn::Tensor v5 = v1[3];
  ::ttnn::Tensor v6 = v1[4];
  ::ttnn::Tensor v7 = v1[5];
  ::ttnn::Tensor v8 = v1[6];
  ::ttnn::Tensor v9 = v1[7];
  ::ttnn::Tensor v10 = v1[8];
  ::ttnn::Tensor v11 = v1[9];
  ::ttnn::Tensor v12 = v1[10];
  ::ttnn::Tensor v13 = v1[11];
  ::ttnn::Tensor v14 = v1[12];
  ::ttnn::Tensor v15 = v1[13];
  ::ttnn::Tensor v16 = v1[14];
  ::ttnn::Tensor v17 = v1[15];
  ::ttnn::Tensor v18 = v1[16];
  ::ttnn::Tensor v19 = v1[17];
  ::ttnn::Tensor v20 = v1[18];
  ::ttnn::Tensor v21 = v1[19];
  ::ttnn::Tensor v22 = v1[20];
  ::ttnn::Tensor v23 = v1[21];
  ::ttnn::Tensor v24 = v1[22];
  ::ttnn::Tensor v25 = v1[23];
  ::ttnn::Tensor v26 = v1[24];
  ::ttnn::Tensor v27 = v1[25];
  ::ttnn::Tensor v28 = v1[26];
  ::ttnn::Tensor v29 = v1[27];
  ::ttnn::Tensor v30 = v1[28];
  ::ttnn::Tensor v31 = v1[29];
  ::ttnn::Tensor v32 = v1[30];
  ::ttnn::Tensor v33 = v1[31];
  ::ttnn::Tensor v34 = v1[32];
  ::ttnn::Tensor v35 = v1[33];
  ::ttnn::Tensor v36 = v1[34];
  ::ttnn::Tensor v37 = v1[35];
  ::ttnn::Tensor v38 = v1[36];
  ::ttnn::Tensor v39 = v1[37];
  ::ttnn::Tensor v40 = v1[38];
  ::ttnn::Tensor v41 = v1[39];
  ::ttnn::Tensor v42 = v1[40];
  ::ttnn::Tensor v43 = v1[41];
  ::ttnn::Tensor v44 = v1[42];
  ::ttnn::Tensor v45 = v1[43];
  ::ttnn::Tensor v46 = v1[44];
  ::ttnn::Tensor v47 = v1[45];
  ::ttnn::Tensor v48 = v1[46];
  ::ttnn::Tensor v49 = v1[47];
  ::ttnn::Tensor v50 = v1[48];
  ::ttnn::Tensor v51 = v1[49];
  ::ttnn::Tensor v52 = v1[50];
  ::ttnn::Tensor v53 = v1[51];
  ::ttnn::Tensor v54 = v1[52];
  ::ttnn::Tensor v55 = v1[53];
  ::ttnn::Tensor v56 = v1[54];
  ::ttnn::Tensor v57 = v1[55];
  ::ttnn::Tensor v58 = v1[56];
  ::ttnn::Tensor v59 = v1[57];
  ::ttnn::Tensor v60 = v1[58];
  ::ttnn::Tensor v61 = v1[59];
  ::ttnn::Tensor v62 = v1[60];
  ::ttnn::Tensor v63 = v1[61];
  ::ttnn::Tensor v64 = v1[62];
  ::ttnn::Tensor v65 = v1[63];
  ::ttnn::Tensor v66 = v1[64];
  ::ttnn::Tensor v67 = v1[65];
  ::ttnn::Tensor v68 = v1[66];
  ::ttnn::Tensor v69 = v1[67];
  ::ttnn::Tensor v70 = v1[68];
  ::ttnn::Tensor v71 = v1[69];
  ::ttnn::Tensor v72 = v1[70];
  ::ttnn::Tensor v73 = v1[71];
  ::ttnn::Tensor v74 = v1[72];
  ::ttnn::Tensor v75 = v1[73];
  ::ttnn::Tensor v76 = v1[74];
  ::ttnn::Tensor v77 = v1[75];
  ::ttnn::Tensor v78 = v1[76];
  ::ttnn::Tensor v79 = v1[77];
  ::ttnn::Tensor v80 = v1[78];
  ::ttnn::Tensor v81 = v1[79];
  ::ttnn::Tensor v82 = v1[80];
  ::ttnn::Tensor v83 = v1[81];
  ::ttnn::Tensor v84 = v1[82];
  ::ttnn::Tensor v85 = v1[83];
  ::ttnn::Tensor v86 = v1[84];
  ::ttnn::Tensor v87 = v1[85];
  ::ttnn::Tensor v88 = v1[86];
  ::ttnn::Tensor v89 = v1[87];
  ::ttnn::Tensor v90 = v1[88];
  ::ttnn::Tensor v91 = v1[89];
  ::ttnn::Tensor v92 = v1[90];
  ::ttnn::Tensor v93 = v1[91];
  ::ttnn::Tensor v94 = v1[92];
  ::ttnn::Tensor v95 = v1[93];
  ::ttnn::Tensor v96 = v1[94];
  ::ttnn::Tensor v97 = v1[95];
  ::ttnn::Tensor v98 = v1[96];
  ::ttnn::Tensor v99 = v1[97];
  ::ttnn::Tensor v100 = v1[98];
  ::ttnn::Tensor v101 = v1[99];
  ::ttnn::Tensor v102 = v1[100];
  ::ttnn::Tensor v103 = v1[101];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v104 = &forward_const_eval_0;
  ::std::vector<::ttnn::Tensor> v105 = util_create_vec(v3);
  ::std::vector<::ttnn::Tensor> *v106 = &g_cached_result_forward_const_eval_0;
  ttnn::constEvalFuncWrapper(v104, v105, v106);
  ::std::vector<::ttnn::Tensor> v107 = g_cached_result_forward_const_eval_0;
  ::ttnn::Tensor v108 = v107[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v109 = &forward_const_eval_1;
  ::std::vector<::ttnn::Tensor> v110 = util_create_vec(v25);
  ::std::vector<::ttnn::Tensor> *v111 = &g_cached_result_forward_const_eval_1;
  ttnn::constEvalFuncWrapper(v109, v110, v111);
  ::std::vector<::ttnn::Tensor> v112 = g_cached_result_forward_const_eval_1;
  ::ttnn::Tensor v113 = v112[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v114 = &forward_const_eval_2;
  ::std::vector<::ttnn::Tensor> v115 = util_create_vec(v29);
  ::std::vector<::ttnn::Tensor> *v116 = &g_cached_result_forward_const_eval_2;
  ttnn::constEvalFuncWrapper(v114, v115, v116);
  ::std::vector<::ttnn::Tensor> v117 = g_cached_result_forward_const_eval_2;
  ::ttnn::Tensor v118 = v117[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v119 = &forward_const_eval_3;
  ::std::vector<::ttnn::Tensor> v120 = util_create_vec(v46);
  ::std::vector<::ttnn::Tensor> *v121 = &g_cached_result_forward_const_eval_3;
  ttnn::constEvalFuncWrapper(v119, v120, v121);
  ::std::vector<::ttnn::Tensor> v122 = g_cached_result_forward_const_eval_3;
  ::ttnn::Tensor v123 = v122[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v124 = &forward_const_eval_4;
  ::std::vector<::ttnn::Tensor> v125 = util_create_vec(v21);
  ::std::vector<::ttnn::Tensor> *v126 = &g_cached_result_forward_const_eval_4;
  ttnn::constEvalFuncWrapper(v124, v125, v126);
  ::std::vector<::ttnn::Tensor> v127 = g_cached_result_forward_const_eval_4;
  ::ttnn::Tensor v128 = v127[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v129 = &forward_const_eval_5;
  ::std::vector<::ttnn::Tensor> v130 = util_create_vec(v32);
  ::std::vector<::ttnn::Tensor> *v131 = &g_cached_result_forward_const_eval_5;
  ttnn::constEvalFuncWrapper(v129, v130, v131);
  ::std::vector<::ttnn::Tensor> v132 = g_cached_result_forward_const_eval_5;
  ::ttnn::Tensor v133 = v132[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v134 = &forward_const_eval_6;
  ::std::vector<::ttnn::Tensor> v135 = util_create_vec(v7);
  ::std::vector<::ttnn::Tensor> *v136 = &g_cached_result_forward_const_eval_6;
  ttnn::constEvalFuncWrapper(v134, v135, v136);
  ::std::vector<::ttnn::Tensor> v137 = g_cached_result_forward_const_eval_6;
  ::ttnn::Tensor v138 = v137[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v139 = &forward_const_eval_7;
  ::std::vector<::ttnn::Tensor> v140 = util_create_vec(v54);
  ::std::vector<::ttnn::Tensor> *v141 = &g_cached_result_forward_const_eval_7;
  ttnn::constEvalFuncWrapper(v139, v140, v141);
  ::std::vector<::ttnn::Tensor> v142 = g_cached_result_forward_const_eval_7;
  ::ttnn::Tensor v143 = v142[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v144 = &forward_const_eval_8;
  ::std::vector<::ttnn::Tensor> v145 = util_create_vec(v40);
  ::std::vector<::ttnn::Tensor> *v146 = &g_cached_result_forward_const_eval_8;
  ttnn::constEvalFuncWrapper(v144, v145, v146);
  ::std::vector<::ttnn::Tensor> v147 = g_cached_result_forward_const_eval_8;
  ::ttnn::Tensor v148 = v147[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v149 = &forward_const_eval_9;
  ::std::vector<::ttnn::Tensor> v150 = util_create_vec(v39);
  ::std::vector<::ttnn::Tensor> *v151 = &g_cached_result_forward_const_eval_9;
  ttnn::constEvalFuncWrapper(v149, v150, v151);
  ::std::vector<::ttnn::Tensor> v152 = g_cached_result_forward_const_eval_9;
  ::ttnn::Tensor v153 = v152[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v154 = &forward_const_eval_10;
  ::std::vector<::ttnn::Tensor> v155 = util_create_vec(v14);
  ::std::vector<::ttnn::Tensor> *v156 = &g_cached_result_forward_const_eval_10;
  ttnn::constEvalFuncWrapper(v154, v155, v156);
  ::std::vector<::ttnn::Tensor> v157 = g_cached_result_forward_const_eval_10;
  ::ttnn::Tensor v158 = v157[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v159 = &forward_const_eval_11;
  ::std::vector<::ttnn::Tensor> v160 = util_create_vec(v15);
  ::std::vector<::ttnn::Tensor> *v161 = &g_cached_result_forward_const_eval_11;
  ttnn::constEvalFuncWrapper(v159, v160, v161);
  ::std::vector<::ttnn::Tensor> v162 = g_cached_result_forward_const_eval_11;
  ::ttnn::Tensor v163 = v162[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v164 = &forward_const_eval_12;
  ::std::vector<::ttnn::Tensor> v165 = util_create_vec(v91);
  ::std::vector<::ttnn::Tensor> *v166 = &g_cached_result_forward_const_eval_12;
  ttnn::constEvalFuncWrapper(v164, v165, v166);
  ::std::vector<::ttnn::Tensor> v167 = g_cached_result_forward_const_eval_12;
  ::ttnn::Tensor v168 = v167[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v169 = &forward_const_eval_13;
  ::std::vector<::ttnn::Tensor> v170 = util_create_vec(v11);
  ::std::vector<::ttnn::Tensor> *v171 = &g_cached_result_forward_const_eval_13;
  ttnn::constEvalFuncWrapper(v169, v170, v171);
  ::std::vector<::ttnn::Tensor> v172 = g_cached_result_forward_const_eval_13;
  ::ttnn::Tensor v173 = v172[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v174 = &forward_const_eval_14;
  ::std::vector<::ttnn::Tensor> v175 = util_create_vec(v18);
  ::std::vector<::ttnn::Tensor> *v176 = &g_cached_result_forward_const_eval_14;
  ttnn::constEvalFuncWrapper(v174, v175, v176);
  ::std::vector<::ttnn::Tensor> v177 = g_cached_result_forward_const_eval_14;
  ::ttnn::Tensor v178 = v177[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v179 = &forward_const_eval_15;
  ::std::vector<::ttnn::Tensor> v180 = util_create_vec(v36);
  ::std::vector<::ttnn::Tensor> *v181 = &g_cached_result_forward_const_eval_15;
  ttnn::constEvalFuncWrapper(v179, v180, v181);
  ::std::vector<::ttnn::Tensor> v182 = g_cached_result_forward_const_eval_15;
  ::ttnn::Tensor v183 = v182[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v184 = &forward_const_eval_16;
  ::std::vector<::ttnn::Tensor> v185 = util_create_vec(v51);
  ::std::vector<::ttnn::Tensor> *v186 = &g_cached_result_forward_const_eval_16;
  ttnn::constEvalFuncWrapper(v184, v185, v186);
  ::std::vector<::ttnn::Tensor> v187 = g_cached_result_forward_const_eval_16;
  ::ttnn::Tensor v188 = v187[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v189 = &forward_const_eval_17;
  ::std::vector<::ttnn::Tensor> v190 = util_create_vec(v6);
  ::std::vector<::ttnn::Tensor> *v191 = &g_cached_result_forward_const_eval_17;
  ttnn::constEvalFuncWrapper(v189, v190, v191);
  ::std::vector<::ttnn::Tensor> v192 = g_cached_result_forward_const_eval_17;
  ::ttnn::Tensor v193 = v192[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v194 = &forward_const_eval_18;
  ::std::vector<::ttnn::Tensor> v195 = util_create_vec(v24);
  ::std::vector<::ttnn::Tensor> *v196 = &g_cached_result_forward_const_eval_18;
  ttnn::constEvalFuncWrapper(v194, v195, v196);
  ::std::vector<::ttnn::Tensor> v197 = g_cached_result_forward_const_eval_18;
  ::ttnn::Tensor v198 = v197[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v199 = &forward_const_eval_19;
  ::std::vector<::ttnn::Tensor> v200 = util_create_vec(v56);
  ::std::vector<::ttnn::Tensor> *v201 = &g_cached_result_forward_const_eval_19;
  ttnn::constEvalFuncWrapper(v199, v200, v201);
  ::std::vector<::ttnn::Tensor> v202 = g_cached_result_forward_const_eval_19;
  ::ttnn::Tensor v203 = v202[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v204 = &forward_const_eval_20;
  ::std::vector<::ttnn::Tensor> v205 = util_create_vec(v49);
  ::std::vector<::ttnn::Tensor> *v206 = &g_cached_result_forward_const_eval_20;
  ttnn::constEvalFuncWrapper(v204, v205, v206);
  ::std::vector<::ttnn::Tensor> v207 = g_cached_result_forward_const_eval_20;
  ::ttnn::Tensor v208 = v207[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v209 = &forward_const_eval_21;
  ::std::vector<::ttnn::Tensor> v210 = util_create_vec(v12);
  ::std::vector<::ttnn::Tensor> *v211 = &g_cached_result_forward_const_eval_21;
  ttnn::constEvalFuncWrapper(v209, v210, v211);
  ::std::vector<::ttnn::Tensor> v212 = g_cached_result_forward_const_eval_21;
  ::ttnn::Tensor v213 = v212[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v214 = &forward_const_eval_22;
  ::std::vector<::ttnn::Tensor> v215 = util_create_vec(v26);
  ::std::vector<::ttnn::Tensor> *v216 = &g_cached_result_forward_const_eval_22;
  ttnn::constEvalFuncWrapper(v214, v215, v216);
  ::std::vector<::ttnn::Tensor> v217 = g_cached_result_forward_const_eval_22;
  ::ttnn::Tensor v218 = v217[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v219 = &forward_const_eval_23;
  ::std::vector<::ttnn::Tensor> v220 = util_create_vec(v71);
  ::std::vector<::ttnn::Tensor> *v221 = &g_cached_result_forward_const_eval_23;
  ttnn::constEvalFuncWrapper(v219, v220, v221);
  ::std::vector<::ttnn::Tensor> v222 = g_cached_result_forward_const_eval_23;
  ::ttnn::Tensor v223 = v222[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v224 = &forward_const_eval_24;
  ::std::vector<::ttnn::Tensor> v225 = util_create_vec(v19);
  ::std::vector<::ttnn::Tensor> *v226 = &g_cached_result_forward_const_eval_24;
  ttnn::constEvalFuncWrapper(v224, v225, v226);
  ::std::vector<::ttnn::Tensor> v227 = g_cached_result_forward_const_eval_24;
  ::ttnn::Tensor v228 = v227[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v229 = &forward_const_eval_25;
  ::std::vector<::ttnn::Tensor> v230 = util_create_vec(v37);
  ::std::vector<::ttnn::Tensor> *v231 = &g_cached_result_forward_const_eval_25;
  ttnn::constEvalFuncWrapper(v229, v230, v231);
  ::std::vector<::ttnn::Tensor> v232 = g_cached_result_forward_const_eval_25;
  ::ttnn::Tensor v233 = v232[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v234 = &forward_const_eval_26;
  ::std::vector<::ttnn::Tensor> v235 = util_create_vec(v38);
  ::std::vector<::ttnn::Tensor> *v236 = &g_cached_result_forward_const_eval_26;
  ttnn::constEvalFuncWrapper(v234, v235, v236);
  ::std::vector<::ttnn::Tensor> v237 = g_cached_result_forward_const_eval_26;
  ::ttnn::Tensor v238 = v237[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v239 = &forward_const_eval_27;
  ::std::vector<::ttnn::Tensor> v240 = util_create_vec(v43);
  ::std::vector<::ttnn::Tensor> *v241 = &g_cached_result_forward_const_eval_27;
  ttnn::constEvalFuncWrapper(v239, v240, v241);
  ::std::vector<::ttnn::Tensor> v242 = g_cached_result_forward_const_eval_27;
  ::ttnn::Tensor v243 = v242[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v244 = &forward_const_eval_28;
  ::std::vector<::ttnn::Tensor> v245 = util_create_vec(v13);
  ::std::vector<::ttnn::Tensor> *v246 = &g_cached_result_forward_const_eval_28;
  ttnn::constEvalFuncWrapper(v244, v245, v246);
  ::std::vector<::ttnn::Tensor> v247 = g_cached_result_forward_const_eval_28;
  ::ttnn::Tensor v248 = v247[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v249 = &forward_const_eval_29;
  ::std::vector<::ttnn::Tensor> v250 = util_create_vec(v31);
  ::std::vector<::ttnn::Tensor> *v251 = &g_cached_result_forward_const_eval_29;
  ttnn::constEvalFuncWrapper(v249, v250, v251);
  ::std::vector<::ttnn::Tensor> v252 = g_cached_result_forward_const_eval_29;
  ::ttnn::Tensor v253 = v252[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v254 = &forward_const_eval_30;
  ::std::vector<::ttnn::Tensor> v255 = util_create_vec(v50);
  ::std::vector<::ttnn::Tensor> *v256 = &g_cached_result_forward_const_eval_30;
  ttnn::constEvalFuncWrapper(v254, v255, v256);
  ::std::vector<::ttnn::Tensor> v257 = g_cached_result_forward_const_eval_30;
  ::ttnn::Tensor v258 = v257[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v259 = &forward_const_eval_31;
  ::std::vector<::ttnn::Tensor> v260 = util_create_vec(v55);
  ::std::vector<::ttnn::Tensor> *v261 = &g_cached_result_forward_const_eval_31;
  ttnn::constEvalFuncWrapper(v259, v260, v261);
  ::std::vector<::ttnn::Tensor> v262 = g_cached_result_forward_const_eval_31;
  ::ttnn::Tensor v263 = v262[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v264 = &forward_const_eval_32;
  ::std::vector<::ttnn::Tensor> v265 = util_create_vec(v5);
  ::std::vector<::ttnn::Tensor> *v266 = &g_cached_result_forward_const_eval_32;
  ttnn::constEvalFuncWrapper(v264, v265, v266);
  ::std::vector<::ttnn::Tensor> v267 = g_cached_result_forward_const_eval_32;
  ::ttnn::Tensor v268 = v267[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v269 = &forward_const_eval_33;
  ::std::vector<::ttnn::Tensor> v270 = util_create_vec(v52);
  ::std::vector<::ttnn::Tensor> *v271 = &g_cached_result_forward_const_eval_33;
  ttnn::constEvalFuncWrapper(v269, v270, v271);
  ::std::vector<::ttnn::Tensor> v272 = g_cached_result_forward_const_eval_33;
  ::ttnn::Tensor v273 = v272[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v274 = &forward_const_eval_34;
  ::std::vector<::ttnn::Tensor> v275 = util_create_vec(v9);
  ::std::vector<::ttnn::Tensor> *v276 = &g_cached_result_forward_const_eval_34;
  ttnn::constEvalFuncWrapper(v274, v275, v276);
  ::std::vector<::ttnn::Tensor> v277 = g_cached_result_forward_const_eval_34;
  ::ttnn::Tensor v278 = v277[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v279 = &forward_const_eval_35;
  ::std::vector<::ttnn::Tensor> v280 = util_create_vec(v23);
  ::std::vector<::ttnn::Tensor> *v281 = &g_cached_result_forward_const_eval_35;
  ttnn::constEvalFuncWrapper(v279, v280, v281);
  ::std::vector<::ttnn::Tensor> v282 = g_cached_result_forward_const_eval_35;
  ::ttnn::Tensor v283 = v282[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v284 = &forward_const_eval_36;
  ::std::vector<::ttnn::Tensor> v285 = util_create_vec(v81);
  ::std::vector<::ttnn::Tensor> *v286 = &g_cached_result_forward_const_eval_36;
  ttnn::constEvalFuncWrapper(v284, v285, v286);
  ::std::vector<::ttnn::Tensor> v287 = g_cached_result_forward_const_eval_36;
  ::ttnn::Tensor v288 = v287[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v289 = &forward_const_eval_37;
  ::std::vector<::ttnn::Tensor> v290 = util_create_vec(v44);
  ::std::vector<::ttnn::Tensor> *v291 = &g_cached_result_forward_const_eval_37;
  ttnn::constEvalFuncWrapper(v289, v290, v291);
  ::std::vector<::ttnn::Tensor> v292 = g_cached_result_forward_const_eval_37;
  ::ttnn::Tensor v293 = v292[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v294 = &forward_const_eval_38;
  ::std::vector<::ttnn::Tensor> v295 = util_create_vec(v42);
  ::std::vector<::ttnn::Tensor> *v296 = &g_cached_result_forward_const_eval_38;
  ttnn::constEvalFuncWrapper(v294, v295, v296);
  ::std::vector<::ttnn::Tensor> v297 = g_cached_result_forward_const_eval_38;
  ::ttnn::Tensor v298 = v297[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v299 = &forward_const_eval_39;
  ::std::vector<::ttnn::Tensor> v300 = util_create_vec(v45);
  ::std::vector<::ttnn::Tensor> *v301 = &g_cached_result_forward_const_eval_39;
  ttnn::constEvalFuncWrapper(v299, v300, v301);
  ::std::vector<::ttnn::Tensor> v302 = g_cached_result_forward_const_eval_39;
  ::ttnn::Tensor v303 = v302[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v304 = &forward_const_eval_40;
  ::std::vector<::ttnn::Tensor> v305 = util_create_vec(v20);
  ::std::vector<::ttnn::Tensor> *v306 = &g_cached_result_forward_const_eval_40;
  ttnn::constEvalFuncWrapper(v304, v305, v306);
  ::std::vector<::ttnn::Tensor> v307 = g_cached_result_forward_const_eval_40;
  ::ttnn::Tensor v308 = v307[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v309 = &forward_const_eval_41;
  ::std::vector<::ttnn::Tensor> v310 = util_create_vec(v48);
  ::std::vector<::ttnn::Tensor> *v311 = &g_cached_result_forward_const_eval_41;
  ttnn::constEvalFuncWrapper(v309, v310, v311);
  ::std::vector<::ttnn::Tensor> v312 = g_cached_result_forward_const_eval_41;
  ::ttnn::Tensor v313 = v312[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v314 = &forward_const_eval_42;
  ::std::vector<::ttnn::Tensor> v315 = util_create_vec(v101);
  ::std::vector<::ttnn::Tensor> *v316 = &g_cached_result_forward_const_eval_42;
  ttnn::constEvalFuncWrapper(v314, v315, v316);
  ::std::vector<::ttnn::Tensor> v317 = g_cached_result_forward_const_eval_42;
  ::ttnn::Tensor v318 = v317[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v319 = &forward_const_eval_43;
  ::std::vector<::ttnn::Tensor> v320 = util_create_vec(v8);
  ::std::vector<::ttnn::Tensor> *v321 = &g_cached_result_forward_const_eval_43;
  ttnn::constEvalFuncWrapper(v319, v320, v321);
  ::std::vector<::ttnn::Tensor> v322 = g_cached_result_forward_const_eval_43;
  ::ttnn::Tensor v323 = v322[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v324 = &forward_const_eval_44;
  ::std::vector<::ttnn::Tensor> v325 = util_create_vec(v27);
  ::std::vector<::ttnn::Tensor> *v326 = &g_cached_result_forward_const_eval_44;
  ttnn::constEvalFuncWrapper(v324, v325, v326);
  ::std::vector<::ttnn::Tensor> v327 = g_cached_result_forward_const_eval_44;
  ::ttnn::Tensor v328 = v327[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v329 = &forward_const_eval_45;
  ::std::vector<::ttnn::Tensor> v330 = util_create_vec(v53);
  ::std::vector<::ttnn::Tensor> *v331 = &g_cached_result_forward_const_eval_45;
  ttnn::constEvalFuncWrapper(v329, v330, v331);
  ::std::vector<::ttnn::Tensor> v332 = g_cached_result_forward_const_eval_45;
  ::ttnn::Tensor v333 = v332[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v334 = &forward_const_eval_46;
  ::std::vector<::ttnn::Tensor> v335 = util_create_vec(v30);
  ::std::vector<::ttnn::Tensor> *v336 = &g_cached_result_forward_const_eval_46;
  ttnn::constEvalFuncWrapper(v334, v335, v336);
  ::std::vector<::ttnn::Tensor> v337 = g_cached_result_forward_const_eval_46;
  ::ttnn::Tensor v338 = v337[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v339 = &forward_const_eval_47;
  ::std::vector<::ttnn::Tensor> v340 = util_create_vec(v4);
  ::std::vector<::ttnn::Tensor> *v341 = &g_cached_result_forward_const_eval_47;
  ttnn::constEvalFuncWrapper(v339, v340, v341);
  ::std::vector<::ttnn::Tensor> v342 = g_cached_result_forward_const_eval_47;
  ::ttnn::Tensor v343 = v342[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v344 = &forward_const_eval_48;
  ::std::vector<::ttnn::Tensor> v345 = util_create_vec(v28);
  ::std::vector<::ttnn::Tensor> *v346 = &g_cached_result_forward_const_eval_48;
  ttnn::constEvalFuncWrapper(v344, v345, v346);
  ::std::vector<::ttnn::Tensor> v347 = g_cached_result_forward_const_eval_48;
  ::ttnn::Tensor v348 = v347[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v349 = &forward_const_eval_49;
  ::std::vector<::ttnn::Tensor> v350 = util_create_vec(v33);
  ::std::vector<::ttnn::Tensor> *v351 = &g_cached_result_forward_const_eval_49;
  ttnn::constEvalFuncWrapper(v349, v350, v351);
  ::std::vector<::ttnn::Tensor> v352 = g_cached_result_forward_const_eval_49;
  ::ttnn::Tensor v353 = v352[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v354 = &forward_const_eval_50;
  ::std::vector<::ttnn::Tensor> v355 = util_create_vec(v47);
  ::std::vector<::ttnn::Tensor> *v356 = &g_cached_result_forward_const_eval_50;
  ttnn::constEvalFuncWrapper(v354, v355, v356);
  ::std::vector<::ttnn::Tensor> v357 = g_cached_result_forward_const_eval_50;
  ::ttnn::Tensor v358 = v357[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v359 = &forward_const_eval_51;
  ::std::vector<::ttnn::Tensor> v360 = util_create_vec(v35);
  ::std::vector<::ttnn::Tensor> *v361 = &g_cached_result_forward_const_eval_51;
  ttnn::constEvalFuncWrapper(v359, v360, v361);
  ::std::vector<::ttnn::Tensor> v362 = g_cached_result_forward_const_eval_51;
  ::ttnn::Tensor v363 = v362[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v364 = &forward_const_eval_52;
  ::std::vector<::ttnn::Tensor> v365 = util_create_vec(v41);
  ::std::vector<::ttnn::Tensor> *v366 = &g_cached_result_forward_const_eval_52;
  ttnn::constEvalFuncWrapper(v364, v365, v366);
  ::std::vector<::ttnn::Tensor> v367 = g_cached_result_forward_const_eval_52;
  ::ttnn::Tensor v368 = v367[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v369 = &forward_const_eval_53;
  ::std::vector<::ttnn::Tensor> v370 = util_create_vec(v16);
  ::std::vector<::ttnn::Tensor> *v371 = &g_cached_result_forward_const_eval_53;
  ttnn::constEvalFuncWrapper(v369, v370, v371);
  ::std::vector<::ttnn::Tensor> v372 = g_cached_result_forward_const_eval_53;
  ::ttnn::Tensor v373 = v372[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v374 = &forward_const_eval_54;
  ::std::vector<::ttnn::Tensor> v375 = util_create_vec(v17);
  ::std::vector<::ttnn::Tensor> *v376 = &g_cached_result_forward_const_eval_54;
  ttnn::constEvalFuncWrapper(v374, v375, v376);
  ::std::vector<::ttnn::Tensor> v377 = g_cached_result_forward_const_eval_54;
  ::ttnn::Tensor v378 = v377[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v379 = &forward_const_eval_55;
  ::std::vector<::ttnn::Tensor> v380 = util_create_vec(v10);
  ::std::vector<::ttnn::Tensor> *v381 = &g_cached_result_forward_const_eval_55;
  ttnn::constEvalFuncWrapper(v379, v380, v381);
  ::std::vector<::ttnn::Tensor> v382 = g_cached_result_forward_const_eval_55;
  ::ttnn::Tensor v383 = v382[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v384 = &forward_const_eval_56;
  ::std::vector<::ttnn::Tensor> v385 = util_create_vec(v22);
  ::std::vector<::ttnn::Tensor> *v386 = &g_cached_result_forward_const_eval_56;
  ttnn::constEvalFuncWrapper(v384, v385, v386);
  ::std::vector<::ttnn::Tensor> v387 = g_cached_result_forward_const_eval_56;
  ::ttnn::Tensor v388 = v387[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v389 = &forward_const_eval_57;
  ::std::vector<::ttnn::Tensor> v390 = util_create_vec(v34);
  ::std::vector<::ttnn::Tensor> *v391 = &g_cached_result_forward_const_eval_57;
  ttnn::constEvalFuncWrapper(v389, v390, v391);
  ::std::vector<::ttnn::Tensor> v392 = g_cached_result_forward_const_eval_57;
  ::ttnn::Tensor v393 = v392[0];
  ttnn::distributed::MeshDevice *v394 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v395 = ttnn::to_layout(
      v2, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v396 = ttnn::permute(
      v395, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v395, false);
  ::ttnn::Tensor v397 = ttnn::reshape(
      v396, ::std::vector<int32_t>{1, 1, 401408, 3},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v396, false);
  ::ttnn::Tensor v398 = ttnn::to_layout(
      v397, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v397, false);
  ::ttnn::Tensor v399 = ::std::get<0>(ttnn::conv2d(
      v398, v57, v394, 3, 64, 8, 224, 224, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v398, false);
  ::ttnn::Tensor v400 = ttnn::multiply(
      v399, v108, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v399, false);
  ::ttnn::Tensor v401 =
      ttnn::add(v400, v343, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v400, false);
  ::ttnn::Tensor v402 = ttnn::relu(
      v401, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v401, false);
  ::ttnn::Tensor v403 = ttnn::to_layout(
      v402, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v402, false);
  ::ttnn::Tensor v404 = ::std::get<0>(ttnn::conv2d(
      v403, v58, v394, 64, 64, 8, 112, 112, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 64, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v403, false);
  ::ttnn::Tensor v405 = ttnn::to_layout(
      v404, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v404, false);
  ::ttnn::Tensor v406 = ::std::get<0>(ttnn::conv2d(
      v405, v59, v394, 64, 64, 8, 112, 112, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v405, false);
  ::ttnn::Tensor v407 = ttnn::multiply(
      v406, v268, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v406, false);
  ::ttnn::Tensor v408 =
      ttnn::add(v407, v193, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v407, false);
  ::ttnn::Tensor v409 = ttnn::relu(
      v408, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v408, false);
  ::ttnn::Tensor v410 = ttnn::to_layout(
      v409, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v409, false);
  ::ttnn::Tensor v411 = ::std::get<0>(ttnn::conv2d(
      v410, v60, v394, 64, 64, 8, 112, 112, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 64, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v410, false);
  ::ttnn::Tensor v412 = ttnn::to_layout(
      v411, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v411, false);
  ::ttnn::Tensor v413 = ::std::get<0>(ttnn::conv2d(
      v412, v61, v394, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v412, false);
  ::ttnn::Tensor v414 = ttnn::multiply(
      v413, v138, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v413, false);
  ::ttnn::Tensor v415 =
      ttnn::add(v414, v323, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v414, false);
  ::ttnn::Tensor v416 = ttnn::relu(
      v415, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v415, false);
  ::ttnn::Tensor v417 = ttnn::to_layout(
      v416, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v418 = ::std::get<0>(ttnn::conv2d(
      v417, v62, v394, 64, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v417, false);
  ::ttnn::Tensor v419 = ttnn::multiply(
      v418, v278, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v418, false);
  ::ttnn::Tensor v420 =
      ttnn::add(v419, v383, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v419, false);
  ::ttnn::Tensor v421 = ttnn::relu(
      v420, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v420, false);
  ::ttnn::Tensor v422 = ttnn::to_layout(
      v421, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v421, false);
  ::ttnn::Tensor v423 = ::std::get<0>(ttnn::conv2d(
      v422, v63, v394, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 128, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v422, false);
  ::ttnn::Tensor v424 = ttnn::to_layout(
      v423, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v423, false);
  ::ttnn::Tensor v425 = ::std::get<0>(ttnn::conv2d(
      v424, v64, v394, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v424, false);
  ::ttnn::Tensor v426 = ttnn::multiply(
      v425, v173, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v425, false);
  ::ttnn::Tensor v427 =
      ttnn::add(v426, v213, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v426, false);
  ::ttnn::Tensor v428 = ttnn::relu(
      v427, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v427, false);
  ::ttnn::Tensor v429 = ttnn::to_layout(
      v428, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v430 = ::std::get<0>(ttnn::conv2d(
      v429, v65, v394, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 128, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v429, false);
  ::ttnn::Tensor v431 = ttnn::to_layout(
      v430, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v430, false);
  ::ttnn::Tensor v432 = ::std::get<0>(ttnn::conv2d(
      v431, v66, v394, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v431, false);
  ::ttnn::Tensor v433 = ttnn::multiply(
      v432, v248, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v432, false);
  ::ttnn::Tensor v434 =
      ttnn::add(v433, v158, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v433, false);
  ::ttnn::Tensor v435 = ttnn::relu(
      v434, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v434, false);
  ::ttnn::Tensor v436 = ttnn::to_layout(
      v435, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v437 = ::std::get<0>(ttnn::conv2d(
      v436, v67, v394, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 128, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v436, false);
  ::ttnn::Tensor v438 = ttnn::to_layout(
      v437, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v437, false);
  ::ttnn::Tensor v439 = ::std::get<0>(ttnn::conv2d(
      v438, v68, v394, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v438, false);
  ::ttnn::Tensor v440 = ttnn::multiply(
      v439, v163, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v439, false);
  ::ttnn::Tensor v441 =
      ttnn::add(v440, v373, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v440, false);
  ::ttnn::Tensor v442 = ttnn::relu(
      v441, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v441, false);
  ::std::vector<::ttnn::Tensor> v443 = util_create_vec(v416, v428, v435, v442);
  ::ttnn::Tensor v444 = ttnn::concat(
      v443, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v442, false);
  ttnn::deallocate(v435, false);
  ttnn::deallocate(v428, false);
  ttnn::deallocate(v416, false);
  ::ttnn::Tensor v445 = ttnn::to_layout(
      v444, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v444, false);
  ::ttnn::Tensor v446 = ::std::get<0>(ttnn::conv2d(
      v445, v69, v394, 448, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v445, false);
  ::ttnn::Tensor v447 = ttnn::multiply(
      v446, v378, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v446, false);
  ::ttnn::Tensor v448 =
      ttnn::add(v447, v178, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v447, false);
  ::ttnn::Tensor v449 = ttnn::relu(
      v448, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v448, false);
  ::ttnn::Tensor v450 = ttnn::reshape(
      v449, ::std::vector<int32_t>{8, 56, 56, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v451 = ttnn::mean(
      v450, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v450, false);
  ::ttnn::Tensor v452 = ttnn::mean(
      v451, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v451, false);
  ::ttnn::Tensor v453 = ttnn::reshape(
      v452, ::std::vector<int32_t>{1, 1, 8, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v452, false);
  ::ttnn::Tensor v454 = ttnn::to_layout(
      v453, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v453, false);
  ::ttnn::Tensor v455 = ::std::get<0>(ttnn::conv2d(
      v454, v70, v394, 256, 256, 8, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v223,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v454, false);
  ::ttnn::Tensor v456 =
      ttnn::add(v455, v228, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v455, false);
  ::ttnn::Tensor v457 = ttnn::relu6(
      v456, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v456, false);
  ::ttnn::Tensor v458 = ttnn::divide(
      v457, v308, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v457, false);
  ::ttnn::Tensor v459 = ttnn::reshape(
      v458, ::std::vector<int32_t>{8, 1, 1, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v458, false);
  ::ttnn::Tensor v460 = ttnn::repeat(v459, ::ttnn::Shape({1, 56, 56, 1}));
  ttnn::deallocate(v459, false);
  ::ttnn::Tensor v461 = ttnn::reshape(
      v460, ::std::vector<int32_t>{1, 1, 25088, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v460, false);
  ::ttnn::Tensor v462 = ttnn::multiply(
      v449, v461, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v461, false);
  ttnn::deallocate(v449, false);
  ::ttnn::Tensor v463 = ttnn::to_layout(
      v462, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v462, false);
  ::std::vector<::ttnn::Tensor> v464 = ttnn::max_pool2d(
      v463, 8, 56, 56, 256, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0},
      ::std::array<uint32_t, 2>{1, 1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v465 = v464[0];
  ttnn::deallocate(v463, false);
  ::ttnn::Tensor v466 = ttnn::to_layout(
      v465, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v467 = ::std::get<0>(ttnn::conv2d(
      v465, v72, v394, 256, 160, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v465, false);
  ::ttnn::Tensor v468 = ttnn::multiply(
      v467, v128, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v467, false);
  ::ttnn::Tensor v469 =
      ttnn::add(v468, v388, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v468, false);
  ::ttnn::Tensor v470 = ttnn::relu(
      v469, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v469, false);
  ::ttnn::Tensor v471 = ttnn::to_layout(
      v470, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v470, false);
  ::ttnn::Tensor v472 = ::std::get<0>(ttnn::conv2d(
      v471, v73, v394, 160, 160, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 160, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v471, false);
  ::ttnn::Tensor v473 = ttnn::to_layout(
      v472, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v472, false);
  ::ttnn::Tensor v474 = ::std::get<0>(ttnn::conv2d(
      v473, v74, v394, 160, 160, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v473, false);
  ::ttnn::Tensor v475 = ttnn::multiply(
      v474, v283, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v474, false);
  ::ttnn::Tensor v476 =
      ttnn::add(v475, v198, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v475, false);
  ::ttnn::Tensor v477 = ttnn::relu(
      v476, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v476, false);
  ::ttnn::Tensor v478 = ttnn::to_layout(
      v477, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v479 = ::std::get<0>(ttnn::conv2d(
      v478, v75, v394, 160, 160, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 160, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v478, false);
  ::ttnn::Tensor v480 = ttnn::to_layout(
      v479, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v479, false);
  ::ttnn::Tensor v481 = ::std::get<0>(ttnn::conv2d(
      v480, v76, v394, 160, 160, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v480, false);
  ::ttnn::Tensor v482 = ttnn::multiply(
      v481, v113, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v481, false);
  ::ttnn::Tensor v483 =
      ttnn::add(v482, v218, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v482, false);
  ::ttnn::Tensor v484 = ttnn::relu(
      v483, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v483, false);
  ::ttnn::Tensor v485 = ttnn::to_layout(
      v484, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v486 = ::std::get<0>(ttnn::conv2d(
      v485, v77, v394, 160, 160, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 160, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v485, false);
  ::ttnn::Tensor v487 = ttnn::to_layout(
      v486, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v486, false);
  ::ttnn::Tensor v488 = ::std::get<0>(ttnn::conv2d(
      v487, v78, v394, 160, 160, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v487, false);
  ::ttnn::Tensor v489 = ttnn::multiply(
      v488, v328, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v488, false);
  ::ttnn::Tensor v490 =
      ttnn::add(v489, v348, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v489, false);
  ::ttnn::Tensor v491 = ttnn::relu(
      v490, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v490, false);
  ::std::vector<::ttnn::Tensor> v492 = util_create_vec(v466, v477, v484, v491);
  ::ttnn::Tensor v493 = ttnn::concat(
      v492, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v491, false);
  ttnn::deallocate(v484, false);
  ttnn::deallocate(v477, false);
  ttnn::deallocate(v466, false);
  ::ttnn::Tensor v494 = ttnn::to_layout(
      v493, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v493, false);
  ::ttnn::Tensor v495 = ::std::get<0>(ttnn::conv2d(
      v494, v79, v394, 736, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v494, false);
  ::ttnn::Tensor v496 = ttnn::multiply(
      v495, v118, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v495, false);
  ::ttnn::Tensor v497 =
      ttnn::add(v496, v338, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v496, false);
  ::ttnn::Tensor v498 = ttnn::relu(
      v497, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v497, false);
  ::ttnn::Tensor v499 = ttnn::reshape(
      v498, ::std::vector<int32_t>{8, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v500 = ttnn::mean(
      v499, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v499, false);
  ::ttnn::Tensor v501 = ttnn::mean(
      v500, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v500, false);
  ::ttnn::Tensor v502 = ttnn::reshape(
      v501, ::std::vector<int32_t>{1, 1, 8, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v501, false);
  ::ttnn::Tensor v503 = ttnn::to_layout(
      v502, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v502, false);
  ::ttnn::Tensor v504 = ::std::get<0>(ttnn::conv2d(
      v503, v80, v394, 512, 512, 8, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v288,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v503, false);
  ::ttnn::Tensor v505 =
      ttnn::add(v504, v253, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v504, false);
  ::ttnn::Tensor v506 = ttnn::relu6(
      v505, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v505, false);
  ::ttnn::Tensor v507 = ttnn::divide(
      v506, v133, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v506, false);
  ::ttnn::Tensor v508 = ttnn::reshape(
      v507, ::std::vector<int32_t>{8, 1, 1, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v507, false);
  ::ttnn::Tensor v509 = ttnn::repeat(v508, ::ttnn::Shape({1, 28, 28, 1}));
  ttnn::deallocate(v508, false);
  ::ttnn::Tensor v510 = ttnn::reshape(
      v509, ::std::vector<int32_t>{1, 1, 6272, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v509, false);
  ::ttnn::Tensor v511 = ttnn::multiply(
      v498, v510, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v510, false);
  ttnn::deallocate(v498, false);
  ::ttnn::Tensor v512 = ttnn::to_layout(
      v511, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v511, false);
  ::std::vector<::ttnn::Tensor> v513 = ttnn::max_pool2d(
      v512, 8, 28, 28, 512, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0},
      ::std::array<uint32_t, 2>{1, 1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v514 = v513[0];
  ttnn::deallocate(v512, false);
  ::ttnn::Tensor v515 = ttnn::to_layout(
      v514, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v516 = ::std::get<0>(ttnn::conv2d(
      v514, v82, v394, 512, 192, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v514, false);
  ::ttnn::Tensor v517 = ttnn::multiply(
      v516, v353, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v516, false);
  ::ttnn::Tensor v518 =
      ttnn::add(v517, v393, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v517, false);
  ::ttnn::Tensor v519 = ttnn::relu(
      v518, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v518, false);
  ::ttnn::Tensor v520 = ttnn::to_layout(
      v519, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v519, false);
  ::ttnn::Tensor v521 = ::std::get<0>(ttnn::conv2d(
      v520, v83, v394, 192, 192, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 192, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v520, false);
  ::ttnn::Tensor v522 = ttnn::to_layout(
      v521, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v521, false);
  ::ttnn::Tensor v523 = ::std::get<0>(ttnn::conv2d(
      v522, v84, v394, 192, 192, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v522, false);
  ::ttnn::Tensor v524 = ttnn::multiply(
      v523, v363, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v523, false);
  ::ttnn::Tensor v525 =
      ttnn::add(v524, v183, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v524, false);
  ::ttnn::Tensor v526 = ttnn::relu(
      v525, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v525, false);
  ::ttnn::Tensor v527 = ttnn::to_layout(
      v526, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v528 = ::std::get<0>(ttnn::conv2d(
      v527, v85, v394, 192, 192, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 192, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v527, false);
  ::ttnn::Tensor v529 = ttnn::to_layout(
      v528, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v528, false);
  ::ttnn::Tensor v530 = ::std::get<0>(ttnn::conv2d(
      v529, v86, v394, 192, 192, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v529, false);
  ::ttnn::Tensor v531 = ttnn::multiply(
      v530, v233, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v530, false);
  ::ttnn::Tensor v532 =
      ttnn::add(v531, v238, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v531, false);
  ::ttnn::Tensor v533 = ttnn::relu(
      v532, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v532, false);
  ::ttnn::Tensor v534 = ttnn::to_layout(
      v533, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v535 = ::std::get<0>(ttnn::conv2d(
      v534, v87, v394, 192, 192, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 192, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v534, false);
  ::ttnn::Tensor v536 = ttnn::to_layout(
      v535, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v535, false);
  ::ttnn::Tensor v537 = ::std::get<0>(ttnn::conv2d(
      v536, v88, v394, 192, 192, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v536, false);
  ::ttnn::Tensor v538 = ttnn::multiply(
      v537, v153, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v537, false);
  ::ttnn::Tensor v539 =
      ttnn::add(v538, v148, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v538, false);
  ::ttnn::Tensor v540 = ttnn::relu(
      v539, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v539, false);
  ::std::vector<::ttnn::Tensor> v541 = util_create_vec(v515, v526, v533, v540);
  ::ttnn::Tensor v542 = ttnn::concat(
      v541, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v540, false);
  ttnn::deallocate(v533, false);
  ttnn::deallocate(v526, false);
  ttnn::deallocate(v515, false);
  ::ttnn::Tensor v543 = ttnn::to_layout(
      v542, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v542, false);
  ::ttnn::Tensor v544 = ::std::get<0>(ttnn::conv2d(
      v543, v89, v394, 1088, 768, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v543, false);
  ::ttnn::Tensor v545 = ttnn::multiply(
      v544, v368, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v544, false);
  ::ttnn::Tensor v546 =
      ttnn::add(v545, v298, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v545, false);
  ::ttnn::Tensor v547 = ttnn::relu(
      v546, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v546, false);
  ::ttnn::Tensor v548 = ttnn::reshape(
      v547, ::std::vector<int32_t>{8, 14, 14, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v549 = ttnn::mean(
      v548, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v548, false);
  ::ttnn::Tensor v550 = ttnn::mean(
      v549, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v549, false);
  ::ttnn::Tensor v551 = ttnn::reshape(
      v550, ::std::vector<int32_t>{1, 1, 8, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v550, false);
  ::ttnn::Tensor v552 = ttnn::to_layout(
      v551, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v551, false);
  ::ttnn::Tensor v553 = ::std::get<0>(ttnn::conv2d(
      v552, v90, v394, 768, 768, 8, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v168,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v552, false);
  ::ttnn::Tensor v554 =
      ttnn::add(v553, v243, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v553, false);
  ::ttnn::Tensor v555 = ttnn::relu6(
      v554, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v554, false);
  ::ttnn::Tensor v556 = ttnn::divide(
      v555, v293, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v555, false);
  ::ttnn::Tensor v557 = ttnn::reshape(
      v556, ::std::vector<int32_t>{8, 1, 1, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v556, false);
  ::ttnn::Tensor v558 = ttnn::repeat(v557, ::ttnn::Shape({1, 14, 14, 1}));
  ttnn::deallocate(v557, false);
  ::ttnn::Tensor v559 = ttnn::reshape(
      v558, ::std::vector<int32_t>{1, 1, 1568, 768},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v558, false);
  ::ttnn::Tensor v560 = ttnn::multiply(
      v547, v559, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v559, false);
  ttnn::deallocate(v547, false);
  ::ttnn::Tensor v561 = ttnn::to_layout(
      v560, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v560, false);
  ::std::vector<::ttnn::Tensor> v562 = ttnn::max_pool2d(
      v561, 8, 14, 14, 768, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0},
      ::std::array<uint32_t, 2>{1, 1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v563 = v562[0];
  ttnn::deallocate(v561, false);
  ::ttnn::Tensor v564 = ttnn::to_layout(
      v563, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v565 = ::std::get<0>(ttnn::conv2d(
      v563, v92, v394, 768, 224, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v563, false);
  ::ttnn::Tensor v566 = ttnn::multiply(
      v565, v303, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v565, false);
  ::ttnn::Tensor v567 =
      ttnn::add(v566, v123, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v566, false);
  ::ttnn::Tensor v568 = ttnn::relu(
      v567, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v567, false);
  ::ttnn::Tensor v569 = ttnn::to_layout(
      v568, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v568, false);
  ::ttnn::Tensor v570 = ::std::get<0>(ttnn::conv2d(
      v569, v93, v394, 224, 224, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 224, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v569, false);
  ::ttnn::Tensor v571 = ttnn::to_layout(
      v570, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v570, false);
  ::ttnn::Tensor v572 = ::std::get<0>(ttnn::conv2d(
      v571, v94, v394, 224, 224, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v571, false);
  ::ttnn::Tensor v573 = ttnn::multiply(
      v572, v358, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v572, false);
  ::ttnn::Tensor v574 =
      ttnn::add(v573, v313, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v573, false);
  ::ttnn::Tensor v575 = ttnn::relu(
      v574, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v574, false);
  ::ttnn::Tensor v576 = ttnn::to_layout(
      v575, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v577 = ::std::get<0>(ttnn::conv2d(
      v576, v95, v394, 224, 224, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 224, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v576, false);
  ::ttnn::Tensor v578 = ttnn::to_layout(
      v577, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v577, false);
  ::ttnn::Tensor v579 = ::std::get<0>(ttnn::conv2d(
      v578, v96, v394, 224, 224, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v578, false);
  ::ttnn::Tensor v580 = ttnn::multiply(
      v579, v208, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v579, false);
  ::ttnn::Tensor v581 =
      ttnn::add(v580, v258, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v580, false);
  ::ttnn::Tensor v582 = ttnn::relu(
      v581, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v581, false);
  ::ttnn::Tensor v583 = ttnn::to_layout(
      v582, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v584 = ::std::get<0>(ttnn::conv2d(
      v583, v97, v394, 224, 224, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 224, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v583, false);
  ::ttnn::Tensor v585 = ttnn::to_layout(
      v584, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v584, false);
  ::ttnn::Tensor v586 = ::std::get<0>(ttnn::conv2d(
      v585, v98, v394, 224, 224, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v585, false);
  ::ttnn::Tensor v587 = ttnn::multiply(
      v586, v188, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v586, false);
  ::ttnn::Tensor v588 =
      ttnn::add(v587, v273, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v587, false);
  ::ttnn::Tensor v589 = ttnn::relu(
      v588, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v588, false);
  ::std::vector<::ttnn::Tensor> v590 = util_create_vec(v564, v575, v582, v589);
  ::ttnn::Tensor v591 = ttnn::concat(
      v590, 3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v589, false);
  ttnn::deallocate(v582, false);
  ttnn::deallocate(v575, false);
  ttnn::deallocate(v564, false);
  ::ttnn::Tensor v592 = ttnn::to_layout(
      v591, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v591, false);
  ::ttnn::Tensor v593 = ::std::get<0>(ttnn::conv2d(
      v592, v99, v394, 1440, 1024, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v592, false);
  ::ttnn::Tensor v594 = ttnn::multiply(
      v593, v333, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v593, false);
  ::ttnn::Tensor v595 =
      ttnn::add(v594, v143, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v594, false);
  ::ttnn::Tensor v596 = ttnn::relu(
      v595, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v595, false);
  ::ttnn::Tensor v597 = ttnn::reshape(
      v596, ::std::vector<int32_t>{8, 7, 7, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v596, false);
  ::ttnn::Tensor v598 = ttnn::permute(
      v597, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v599 = ttnn::mean(
      v597, ::ttsl::SmallVector<int32_t>{1}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v597, false);
  ::ttnn::Tensor v600 = ttnn::mean(
      v599, ::ttsl::SmallVector<int32_t>{2}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v599, false);
  ::ttnn::Tensor v601 = ttnn::reshape(
      v600, ::std::vector<int32_t>{1, 1, 8, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v600, false);
  ::ttnn::Tensor v602 = ttnn::to_layout(
      v601, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v601, false);
  ::ttnn::Tensor v603 = ::std::get<0>(ttnn::conv2d(
      v602, v100, v394, 1024, 1024, 8, 1, 1, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT16, v318,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v602, false);
  ::ttnn::Tensor v604 =
      ttnn::add(v603, v263, ::ttnn::DataType::BFLOAT16,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v603, false);
  ::ttnn::Tensor v605 = ttnn::relu6(
      v604, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                 ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v604, false);
  ::ttnn::Tensor v606 = ttnn::divide(
      v605, v203, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v605, false);
  ::ttnn::Tensor v607 = ttnn::reshape(
      v606, ::std::vector<int32_t>{8, 1, 1, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v606, false);
  ::ttnn::Tensor v608 = ttnn::permute(
      v607, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v607, false);
  ::ttnn::Tensor v609 = ttnn::reshape(
      v608, ::std::vector<int32_t>{8, 1, 1024, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v608, false);
  ::ttnn::Tensor v610 = ttnn::reshape(
      v598, ::std::vector<int32_t>{8, 1, 1024, 49},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v598, false);
  ::ttnn::Tensor v611 = ttnn::multiply(
      v610, v609, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v610, false);
  ttnn::deallocate(v609, false);
  ::ttnn::Tensor v612 = ttnn::mean(
      v611, ::ttsl::SmallVector<int32_t>{3}, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v611, false);
  ::ttnn::Tensor v613 = ttnn::reshape(
      v612, ::std::vector<int32_t>{8, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v612, false);
  ::ttnn::Tensor v614 = ttnn::linear(
      v613, v102, v103, false, false,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::DataType::BFLOAT16, ::std::nullopt, ::std::nullopt,
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::MathFidelity::HiFi4, .fp32_dest_acc_en = true});
  ttnn::deallocate(v613, false);
  ::std::vector<::ttnn::Tensor> v615 = util_create_vec(v614);
  return v615;
}
::std::vector<::ttnn::Tensor> create_inputs_for_forward() {
  ttnn::distributed::MeshDevice *v1 = ttnn::DeviceGetter::getInstance();
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v2 = *v1;
  ::ttnn::Tensor v3 = ttnn::ones(
      ::ttnn::Shape({8, 3, 224, 224}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, v2,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::ones(
      ::ttnn::Shape({1, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::ones(
      ::ttnn::Shape({1, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::ones(
      ::ttnn::Shape({1, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::ones(
      ::ttnn::Shape({1, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::ones(
      ::ttnn::Shape({1, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::ones(
      ::ttnn::Shape({1, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::ones(
      ::ttnn::Shape({1, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::ones(
      ::ttnn::Shape({1, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::ones(
      ::ttnn::Shape({1, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::ones(
      ::ttnn::Shape({1, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::ones(
      ::ttnn::Shape({1, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::ones(
      ::ttnn::Shape({1, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::ones(
      ::ttnn::Shape({1, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v57 = ttnn::ones(
      ::ttnn::Shape({1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v58 = ttnn::ones(
      ::ttnn::Shape({64, 3, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v59 = ttnn::ones(
      ::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v60 = ttnn::ones(
      ::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v61 = ttnn::ones(
      ::ttnn::Shape({64, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v62 = ttnn::ones(
      ::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v63 = ttnn::ones(
      ::ttnn::Shape({128, 64, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v64 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v65 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v66 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v67 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v68 = ttnn::ones(
      ::ttnn::Shape({128, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v69 = ttnn::ones(
      ::ttnn::Shape({128, 128, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v70 = ttnn::ones(
      ::ttnn::Shape({256, 448, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v71 = ttnn::ones(
      ::ttnn::Shape({256, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v72 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v73 = ttnn::ones(
      ::ttnn::Shape({160, 256, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v74 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v75 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v76 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v77 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v78 = ttnn::ones(
      ::ttnn::Shape({160, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v79 = ttnn::ones(
      ::ttnn::Shape({160, 160, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v80 = ttnn::ones(
      ::ttnn::Shape({512, 736, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v81 = ttnn::ones(
      ::ttnn::Shape({512, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v82 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v83 = ttnn::ones(
      ::ttnn::Shape({192, 512, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v84 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v85 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v86 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v87 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v88 = ttnn::ones(
      ::ttnn::Shape({192, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v89 = ttnn::ones(
      ::ttnn::Shape({192, 192, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v90 = ttnn::ones(
      ::ttnn::Shape({768, 1088, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v91 = ttnn::ones(
      ::ttnn::Shape({768, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v92 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 768}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v93 = ttnn::ones(
      ::ttnn::Shape({224, 768, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v94 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v95 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v96 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v97 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v98 = ttnn::ones(
      ::ttnn::Shape({224, 1, 3, 3}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v99 = ttnn::ones(
      ::ttnn::Shape({224, 224, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v100 = ttnn::ones(
      ::ttnn::Shape({1024, 1440, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v101 = ttnn::ones(
      ::ttnn::Shape({1024, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v102 = ttnn::ones(
      ::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v103 = *v1;
  ::ttnn::Tensor v104 = ttnn::ones(
      ::ttnn::Shape({1024, 1000}), ::ttnn::DataType::BFLOAT16,
      ::ttnn::Layout::TILE, v103,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>>
      v105 = *v1;
  ::ttnn::Tensor v106 = ttnn::ones(
      ::ttnn::Shape({1000}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE,
      v105,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v107 = util_create_vec(
      v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
      v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33,
      v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48,
      v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63,
      v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78,
      v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93,
      v94, v95, v96, v97, v98, v99, v100, v101, v102, v104, v106);
  return v107;
}
int32_t main() {
  ::std::vector<::ttnn::Tensor> v1 = create_inputs_for_forward();

  // Get device and enable program cache
  ttnn::distributed::MeshDevice *device = ttnn::DeviceGetter::getInstance();
  device->enable_program_cache();

  // Warmup run (kernel compilation + program cache population)
  std::cout << "Warmup run 1 (kernel compilation)..." << std::endl;
  ::std::vector<::ttnn::Tensor> v_w1 = forward(v1);
  ::ttnn::Tensor t_w1 = ttnn::from_device(v_w1[0]);

  std::cout << "Warmup run 2 (program cache hit)..." << std::endl;
  ::std::vector<::ttnn::Tensor> v_w2 = forward(v1);
  ::ttnn::Tensor t_w2 = ttnn::from_device(v_w2[0]);

  std::cout << "Program cache entries: " << device->num_program_cache_entries() << std::endl;

  // Benchmark loop
  int num_iters = 10;
  int batch_size = 8;
  std::cout << "\n=== Benchmark (program cache enabled) ===" << std::endl;
  for (int i = 0; i < num_iters; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    ::std::vector<::ttnn::Tensor> v2 = forward(v1);
    ::ttnn::Tensor t = ttnn::from_device(v2[0]);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double fps = batch_size / duration.count();
    std::cout << "Iter " << i << ": " << duration.count() << "s, " << fps << " FPS" << std::endl;
  }

  return 0;
}
