// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#include "operations/trace.hpp"
#include "ttnn/common/queue_id.hpp"
#include <chrono>
template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

#include "ttnn-precompiled.hpp"
::std::vector<::ttnn::Tensor> forward_const_eval_0(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_1(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_2(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_3(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v4 = *v3;
  ::ttnn::Tensor v5 = ttnn::zeros(::ttnn::Shape({8, 2048, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::add(v2, v5, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_4(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_5(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_6(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_7(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v4 = *v3;
  ::ttnn::Tensor v5 = ttnn::zeros(::ttnn::Shape({8, 2048, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::add(v2, v5, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_8(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_9(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_10(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_11(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_12(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_13(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_14(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 3, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 3, 64, 8, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{3, 3, 3, 3}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_15(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_16(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_17(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_18(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_19(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_20(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 256, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_21(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_22(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_23(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_24(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v2, ::std::vector<int32_t>{1, 1000}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_25(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_26(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 1024, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_27(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v2, ::std::vector<int32_t>{1, 1, 2048, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::transpose(v4, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v3;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({8, 1, 49, 2048}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v9 = util_create_vec(v8);
  return v9;
}

::std::vector<::ttnn::Tensor> forward_const_eval_28(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_29(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_30(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_31(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_32(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_33(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_34(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_35(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_36(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_37(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_38(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_39(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_40(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_41(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_42(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v2, ::std::vector<int32_t>{1, 1, 2048, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::transpose(v4, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::std::vector<::ttnn::Tensor> v6 = util_create_vec(v5);
  return v6;
}

::std::vector<::ttnn::Tensor> forward_const_eval_43(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 1024, 2048, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_44(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_45(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_46(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_47(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_48(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{3136, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 512, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_49(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_50(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_51(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{512, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_52(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_53(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_54(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_55(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 512, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_56(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_57(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v6 = *v4;
  ::ttnn::Tensor v7 = ttnn::zeros(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v6, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::add(v5, v7, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v4);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v8, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v12 = ttnn::to_layout(v11, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::from_device(v12);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::operations::conv::conv2d::prepare_conv_weights(v13, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v13, false);
  ::std::vector<::ttnn::Tensor> v15 = util_create_vec(v14);
  return v15;
}

::std::vector<::ttnn::Tensor> forward_const_eval_58(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::std::optional<::std::reference_wrapper<::ttnn::distributed::MeshDevice>> v4 = *v3;
  ::ttnn::Tensor v5 = ttnn::zeros(::ttnn::Shape({8, 2048, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::add(v2, v5, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
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
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_58;
::std::vector<::ttnn::Tensor> forward(const ::std::vector<::ttnn::Tensor> &v1) {
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
  ::ttnn::Tensor v104 = v1[102];
  ::ttnn::Tensor v105 = v1[103];
  ::ttnn::Tensor v106 = v1[104];
  ::ttnn::Tensor v107 = v1[105];
  ::ttnn::Tensor v108 = v1[106];
  ::ttnn::Tensor v109 = v1[107];
  ::ttnn::Tensor v110 = v1[108];
  ::ttnn::Tensor v111 = v1[109];
  ::ttnn::Tensor v112 = v1[110];
  ::ttnn::Tensor v113 = v1[111];
  ::ttnn::Tensor v114 = v1[112];
  ::ttnn::Tensor v115 = v1[113];
  ::ttnn::Tensor v116 = v1[114];
  ::ttnn::Tensor v117 = v1[115];
  ::ttnn::Tensor v118 = v1[116];
  ::ttnn::Tensor v119 = v1[117];
  ::ttnn::Tensor v120 = v1[118];
  ::ttnn::Tensor v121 = v1[119];
  ::ttnn::Tensor v122 = v1[120];
  ::ttnn::Tensor v123 = v1[121];
  ::ttnn::Tensor v124 = v1[122];
  ::ttnn::Tensor v125 = v1[123];
  ::ttnn::Tensor v126 = v1[124];
  ::ttnn::Tensor v127 = v1[125];
  ::ttnn::Tensor v128 = v1[126];
  ::ttnn::Tensor v129 = v1[127];
  ::ttnn::Tensor v130 = v1[128];
  ::ttnn::Tensor v131 = v1[129];
  ::ttnn::Tensor v132 = v1[130];
  ::ttnn::Tensor v133 = v1[131];
  ::ttnn::Tensor v134 = v1[132];
  ::ttnn::Tensor v135 = v1[133];
  ::ttnn::Tensor v136 = v1[134];
  ::ttnn::Tensor v137 = v1[135];
  ::ttnn::Tensor v138 = v1[136];
  ::ttnn::Tensor v139 = v1[137];
  ::ttnn::Tensor v140 = v1[138];
  ::ttnn::Tensor v141 = v1[139];
  ::ttnn::Tensor v142 = v1[140];
  ::ttnn::Tensor v143 = v1[141];
  ::ttnn::Tensor v144 = v1[142];
  ::ttnn::Tensor v145 = v1[143];
  ::ttnn::Tensor v146 = v1[144];
  ::ttnn::Tensor v147 = v1[145];
  ::ttnn::Tensor v148 = v1[146];
  ::ttnn::Tensor v149 = v1[147];
  ::ttnn::Tensor v150 = v1[148];
  ::ttnn::Tensor v151 = v1[149];
  ::ttnn::Tensor v152 = v1[150];
  ::ttnn::Tensor v153 = v1[151];
  ::ttnn::Tensor v154 = v1[152];
  ::ttnn::Tensor v155 = v1[153];
  ::ttnn::Tensor v156 = v1[154];
  ::ttnn::Tensor v157 = v1[155];
  ::ttnn::Tensor v158 = v1[156];
  ::ttnn::Tensor v159 = v1[157];
  ::ttnn::Tensor v160 = v1[158];
  ::ttnn::Tensor v161 = v1[159];
  ::ttnn::Tensor v162 = v1[160];
  ::ttnn::Tensor v163 = v1[161];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v164 = &forward_const_eval_0;
  ::std::vector<::ttnn::Tensor> v165 = util_create_vec(v87, v151);
  ::std::vector<::ttnn::Tensor>* v166 = &g_cached_result_forward_const_eval_0;
  ttnn::constEvalFuncWrapper(v164, v165, v166);
  ::std::vector<::ttnn::Tensor> v167 = g_cached_result_forward_const_eval_0;
  ::ttnn::Tensor v168 = v167[0];
  ttnn::deallocate(v151, false);
  ttnn::deallocate(v87, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v169 = &forward_const_eval_1;
  ::std::vector<::ttnn::Tensor> v170 = util_create_vec(v49, v132);
  ::std::vector<::ttnn::Tensor>* v171 = &g_cached_result_forward_const_eval_1;
  ttnn::constEvalFuncWrapper(v169, v170, v171);
  ::std::vector<::ttnn::Tensor> v172 = g_cached_result_forward_const_eval_1;
  ::ttnn::Tensor v173 = v172[0];
  ttnn::deallocate(v132, false);
  ttnn::deallocate(v49, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v174 = &forward_const_eval_2;
  ::std::vector<::ttnn::Tensor> v175 = util_create_vec(v37, v126);
  ::std::vector<::ttnn::Tensor>* v176 = &g_cached_result_forward_const_eval_2;
  ttnn::constEvalFuncWrapper(v174, v175, v176);
  ::std::vector<::ttnn::Tensor> v177 = g_cached_result_forward_const_eval_2;
  ::ttnn::Tensor v178 = v177[0];
  ttnn::deallocate(v126, false);
  ttnn::deallocate(v37, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v179 = &forward_const_eval_3;
  ::std::vector<::ttnn::Tensor> v180 = util_create_vec(v101);
  ::std::vector<::ttnn::Tensor>* v181 = &g_cached_result_forward_const_eval_3;
  ttnn::constEvalFuncWrapper(v179, v180, v181);
  ::std::vector<::ttnn::Tensor> v182 = g_cached_result_forward_const_eval_3;
  ::ttnn::Tensor v183 = v182[0];
  ttnn::deallocate(v101, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v184 = &forward_const_eval_4;
  ::std::vector<::ttnn::Tensor> v185 = util_create_vec(v67, v141);
  ::std::vector<::ttnn::Tensor>* v186 = &g_cached_result_forward_const_eval_4;
  ttnn::constEvalFuncWrapper(v184, v185, v186);
  ::std::vector<::ttnn::Tensor> v187 = g_cached_result_forward_const_eval_4;
  ::ttnn::Tensor v188 = v187[0];
  ttnn::deallocate(v141, false);
  ttnn::deallocate(v67, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v189 = &forward_const_eval_5;
  ::std::vector<::ttnn::Tensor> v190 = util_create_vec(v15, v115);
  ::std::vector<::ttnn::Tensor>* v191 = &g_cached_result_forward_const_eval_5;
  ttnn::constEvalFuncWrapper(v189, v190, v191);
  ::std::vector<::ttnn::Tensor> v192 = g_cached_result_forward_const_eval_5;
  ::ttnn::Tensor v193 = v192[0];
  ttnn::deallocate(v115, false);
  ttnn::deallocate(v15, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v194 = &forward_const_eval_6;
  ::std::vector<::ttnn::Tensor> v195 = util_create_vec(v154);
  ::std::vector<::ttnn::Tensor>* v196 = &g_cached_result_forward_const_eval_6;
  ttnn::constEvalFuncWrapper(v194, v195, v196);
  ::std::vector<::ttnn::Tensor> v197 = g_cached_result_forward_const_eval_6;
  ::ttnn::Tensor v198 = v197[0];
  ttnn::deallocate(v154, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v199 = &forward_const_eval_7;
  ::std::vector<::ttnn::Tensor> v200 = util_create_vec(v93);
  ::std::vector<::ttnn::Tensor>* v201 = &g_cached_result_forward_const_eval_7;
  ttnn::constEvalFuncWrapper(v199, v200, v201);
  ::std::vector<::ttnn::Tensor> v202 = g_cached_result_forward_const_eval_7;
  ::ttnn::Tensor v203 = v202[0];
  ttnn::deallocate(v93, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v204 = &forward_const_eval_8;
  ::std::vector<::ttnn::Tensor> v205 = util_create_vec(v11, v113);
  ::std::vector<::ttnn::Tensor>* v206 = &g_cached_result_forward_const_eval_8;
  ttnn::constEvalFuncWrapper(v204, v205, v206);
  ::std::vector<::ttnn::Tensor> v207 = g_cached_result_forward_const_eval_8;
  ::ttnn::Tensor v208 = v207[0];
  ttnn::deallocate(v113, false);
  ttnn::deallocate(v11, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v209 = &forward_const_eval_9;
  ::std::vector<::ttnn::Tensor> v210 = util_create_vec(v33, v124);
  ::std::vector<::ttnn::Tensor>* v211 = &g_cached_result_forward_const_eval_9;
  ttnn::constEvalFuncWrapper(v209, v210, v211);
  ::std::vector<::ttnn::Tensor> v212 = g_cached_result_forward_const_eval_9;
  ::ttnn::Tensor v213 = v212[0];
  ttnn::deallocate(v124, false);
  ttnn::deallocate(v33, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v214 = &forward_const_eval_10;
  ::std::vector<::ttnn::Tensor> v215 = util_create_vec(v43, v129);
  ::std::vector<::ttnn::Tensor>* v216 = &g_cached_result_forward_const_eval_10;
  ttnn::constEvalFuncWrapper(v214, v215, v216);
  ::std::vector<::ttnn::Tensor> v217 = g_cached_result_forward_const_eval_10;
  ::ttnn::Tensor v218 = v217[0];
  ttnn::deallocate(v129, false);
  ttnn::deallocate(v43, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v219 = &forward_const_eval_11;
  ::std::vector<::ttnn::Tensor> v220 = util_create_vec(v63, v139);
  ::std::vector<::ttnn::Tensor>* v221 = &g_cached_result_forward_const_eval_11;
  ttnn::constEvalFuncWrapper(v219, v220, v221);
  ::std::vector<::ttnn::Tensor> v222 = g_cached_result_forward_const_eval_11;
  ::ttnn::Tensor v223 = v222[0];
  ttnn::deallocate(v139, false);
  ttnn::deallocate(v63, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v224 = &forward_const_eval_12;
  ::std::vector<::ttnn::Tensor> v225 = util_create_vec(v91, v153);
  ::std::vector<::ttnn::Tensor>* v226 = &g_cached_result_forward_const_eval_12;
  ttnn::constEvalFuncWrapper(v224, v225, v226);
  ::std::vector<::ttnn::Tensor> v227 = g_cached_result_forward_const_eval_12;
  ::ttnn::Tensor v228 = v227[0];
  ttnn::deallocate(v153, false);
  ttnn::deallocate(v91, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v229 = &forward_const_eval_13;
  ::std::vector<::ttnn::Tensor> v230 = util_create_vec(v75, v145);
  ::std::vector<::ttnn::Tensor>* v231 = &g_cached_result_forward_const_eval_13;
  ttnn::constEvalFuncWrapper(v229, v230, v231);
  ::std::vector<::ttnn::Tensor> v232 = g_cached_result_forward_const_eval_13;
  ::ttnn::Tensor v233 = v232[0];
  ttnn::deallocate(v145, false);
  ttnn::deallocate(v75, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v234 = &forward_const_eval_14;
  ::std::vector<::ttnn::Tensor> v235 = util_create_vec(v3, v109);
  ::std::vector<::ttnn::Tensor>* v236 = &g_cached_result_forward_const_eval_14;
  ttnn::constEvalFuncWrapper(v234, v235, v236);
  ::std::vector<::ttnn::Tensor> v237 = g_cached_result_forward_const_eval_14;
  ::ttnn::Tensor v238 = v237[0];
  ttnn::deallocate(v109, false);
  ttnn::deallocate(v3, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v239 = &forward_const_eval_15;
  ::std::vector<::ttnn::Tensor> v240 = util_create_vec(v83, v149);
  ::std::vector<::ttnn::Tensor>* v241 = &g_cached_result_forward_const_eval_15;
  ttnn::constEvalFuncWrapper(v239, v240, v241);
  ::std::vector<::ttnn::Tensor> v242 = g_cached_result_forward_const_eval_15;
  ::ttnn::Tensor v243 = v242[0];
  ttnn::deallocate(v149, false);
  ttnn::deallocate(v83, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v244 = &forward_const_eval_16;
  ::std::vector<::ttnn::Tensor> v245 = util_create_vec(v55, v135);
  ::std::vector<::ttnn::Tensor>* v246 = &g_cached_result_forward_const_eval_16;
  ttnn::constEvalFuncWrapper(v244, v245, v246);
  ::std::vector<::ttnn::Tensor> v247 = g_cached_result_forward_const_eval_16;
  ::ttnn::Tensor v248 = v247[0];
  ttnn::deallocate(v135, false);
  ttnn::deallocate(v55, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v249 = &forward_const_eval_17;
  ::std::vector<::ttnn::Tensor> v250 = util_create_vec(v23, v119);
  ::std::vector<::ttnn::Tensor>* v251 = &g_cached_result_forward_const_eval_17;
  ttnn::constEvalFuncWrapper(v249, v250, v251);
  ::std::vector<::ttnn::Tensor> v252 = g_cached_result_forward_const_eval_17;
  ::ttnn::Tensor v253 = v252[0];
  ttnn::deallocate(v119, false);
  ttnn::deallocate(v23, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v254 = &forward_const_eval_18;
  ::std::vector<::ttnn::Tensor> v255 = util_create_vec(v21, v118);
  ::std::vector<::ttnn::Tensor>* v256 = &g_cached_result_forward_const_eval_18;
  ttnn::constEvalFuncWrapper(v254, v255, v256);
  ::std::vector<::ttnn::Tensor> v257 = g_cached_result_forward_const_eval_18;
  ::ttnn::Tensor v258 = v257[0];
  ttnn::deallocate(v118, false);
  ttnn::deallocate(v21, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v259 = &forward_const_eval_19;
  ::std::vector<::ttnn::Tensor> v260 = util_create_vec(v41, v128);
  ::std::vector<::ttnn::Tensor>* v261 = &g_cached_result_forward_const_eval_19;
  ttnn::constEvalFuncWrapper(v259, v260, v261);
  ::std::vector<::ttnn::Tensor> v262 = g_cached_result_forward_const_eval_19;
  ::ttnn::Tensor v263 = v262[0];
  ttnn::deallocate(v128, false);
  ttnn::deallocate(v41, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v264 = &forward_const_eval_20;
  ::std::vector<::ttnn::Tensor> v265 = util_create_vec(v51, v133);
  ::std::vector<::ttnn::Tensor>* v266 = &g_cached_result_forward_const_eval_20;
  ttnn::constEvalFuncWrapper(v264, v265, v266);
  ::std::vector<::ttnn::Tensor> v267 = g_cached_result_forward_const_eval_20;
  ::ttnn::Tensor v268 = v267[0];
  ttnn::deallocate(v133, false);
  ttnn::deallocate(v51, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v269 = &forward_const_eval_21;
  ::std::vector<::ttnn::Tensor> v270 = util_create_vec(v9, v112);
  ::std::vector<::ttnn::Tensor>* v271 = &g_cached_result_forward_const_eval_21;
  ttnn::constEvalFuncWrapper(v269, v270, v271);
  ::std::vector<::ttnn::Tensor> v272 = g_cached_result_forward_const_eval_21;
  ::ttnn::Tensor v273 = v272[0];
  ttnn::deallocate(v112, false);
  ttnn::deallocate(v9, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v274 = &forward_const_eval_22;
  ::std::vector<::ttnn::Tensor> v275 = util_create_vec(v19, v117);
  ::std::vector<::ttnn::Tensor>* v276 = &g_cached_result_forward_const_eval_22;
  ttnn::constEvalFuncWrapper(v274, v275, v276);
  ::std::vector<::ttnn::Tensor> v277 = g_cached_result_forward_const_eval_22;
  ::ttnn::Tensor v278 = v277[0];
  ttnn::deallocate(v117, false);
  ttnn::deallocate(v19, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v279 = &forward_const_eval_23;
  ::std::vector<::ttnn::Tensor> v280 = util_create_vec(v61, v138);
  ::std::vector<::ttnn::Tensor>* v281 = &g_cached_result_forward_const_eval_23;
  ttnn::constEvalFuncWrapper(v279, v280, v281);
  ::std::vector<::ttnn::Tensor> v282 = g_cached_result_forward_const_eval_23;
  ::ttnn::Tensor v283 = v282[0];
  ttnn::deallocate(v138, false);
  ttnn::deallocate(v61, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v284 = &forward_const_eval_24;
  ::std::vector<::ttnn::Tensor> v285 = util_create_vec(v163);
  ::std::vector<::ttnn::Tensor>* v286 = &g_cached_result_forward_const_eval_24;
  ttnn::constEvalFuncWrapper(v284, v285, v286);
  ::std::vector<::ttnn::Tensor> v287 = g_cached_result_forward_const_eval_24;
  ::ttnn::Tensor v288 = v287[0];
  ttnn::deallocate(v163, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v289 = &forward_const_eval_25;
  ::std::vector<::ttnn::Tensor> v290 = util_create_vec(v85, v150);
  ::std::vector<::ttnn::Tensor>* v291 = &g_cached_result_forward_const_eval_25;
  ttnn::constEvalFuncWrapper(v289, v290, v291);
  ::std::vector<::ttnn::Tensor> v292 = g_cached_result_forward_const_eval_25;
  ::ttnn::Tensor v293 = v292[0];
  ttnn::deallocate(v150, false);
  ttnn::deallocate(v85, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v294 = &forward_const_eval_26;
  ::std::vector<::ttnn::Tensor> v295 = util_create_vec(v57, v136);
  ::std::vector<::ttnn::Tensor>* v296 = &g_cached_result_forward_const_eval_26;
  ttnn::constEvalFuncWrapper(v294, v295, v296);
  ::std::vector<::ttnn::Tensor> v297 = g_cached_result_forward_const_eval_26;
  ::ttnn::Tensor v298 = v297[0];
  ttnn::deallocate(v136, false);
  ttnn::deallocate(v57, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v299 = &forward_const_eval_27;
  ::std::vector<::ttnn::Tensor> v300 = util_create_vec(v107);
  ::std::vector<::ttnn::Tensor>* v301 = &g_cached_result_forward_const_eval_27;
  ttnn::constEvalFuncWrapper(v299, v300, v301);
  ::std::vector<::ttnn::Tensor> v302 = g_cached_result_forward_const_eval_27;
  ::ttnn::Tensor v303 = v302[0];
  ttnn::deallocate(v107, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v304 = &forward_const_eval_28;
  ::std::vector<::ttnn::Tensor> v305 = util_create_vec(v158);
  ::std::vector<::ttnn::Tensor>* v306 = &g_cached_result_forward_const_eval_28;
  ttnn::constEvalFuncWrapper(v304, v305, v306);
  ::std::vector<::ttnn::Tensor> v307 = g_cached_result_forward_const_eval_28;
  ::ttnn::Tensor v308 = v307[0];
  ttnn::deallocate(v158, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v309 = &forward_const_eval_29;
  ::std::vector<::ttnn::Tensor> v310 = util_create_vec(v81, v148);
  ::std::vector<::ttnn::Tensor>* v311 = &g_cached_result_forward_const_eval_29;
  ttnn::constEvalFuncWrapper(v309, v310, v311);
  ::std::vector<::ttnn::Tensor> v312 = g_cached_result_forward_const_eval_29;
  ::ttnn::Tensor v313 = v312[0];
  ttnn::deallocate(v148, false);
  ttnn::deallocate(v81, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v314 = &forward_const_eval_30;
  ::std::vector<::ttnn::Tensor> v315 = util_create_vec(v13, v114);
  ::std::vector<::ttnn::Tensor>* v316 = &g_cached_result_forward_const_eval_30;
  ttnn::constEvalFuncWrapper(v314, v315, v316);
  ::std::vector<::ttnn::Tensor> v317 = g_cached_result_forward_const_eval_30;
  ::ttnn::Tensor v318 = v317[0];
  ttnn::deallocate(v114, false);
  ttnn::deallocate(v13, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v319 = &forward_const_eval_31;
  ::std::vector<::ttnn::Tensor> v320 = util_create_vec(v53, v134);
  ::std::vector<::ttnn::Tensor>* v321 = &g_cached_result_forward_const_eval_31;
  ttnn::constEvalFuncWrapper(v319, v320, v321);
  ::std::vector<::ttnn::Tensor> v322 = g_cached_result_forward_const_eval_31;
  ::ttnn::Tensor v323 = v322[0];
  ttnn::deallocate(v134, false);
  ttnn::deallocate(v53, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v324 = &forward_const_eval_32;
  ::std::vector<::ttnn::Tensor> v325 = util_create_vec(v103, v159);
  ::std::vector<::ttnn::Tensor>* v326 = &g_cached_result_forward_const_eval_32;
  ttnn::constEvalFuncWrapper(v324, v325, v326);
  ::std::vector<::ttnn::Tensor> v327 = g_cached_result_forward_const_eval_32;
  ::ttnn::Tensor v328 = v327[0];
  ttnn::deallocate(v159, false);
  ttnn::deallocate(v103, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v329 = &forward_const_eval_33;
  ::std::vector<::ttnn::Tensor> v330 = util_create_vec(v97, v156);
  ::std::vector<::ttnn::Tensor>* v331 = &g_cached_result_forward_const_eval_33;
  ttnn::constEvalFuncWrapper(v329, v330, v331);
  ::std::vector<::ttnn::Tensor> v332 = g_cached_result_forward_const_eval_33;
  ::ttnn::Tensor v333 = v332[0];
  ttnn::deallocate(v156, false);
  ttnn::deallocate(v97, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v334 = &forward_const_eval_34;
  ::std::vector<::ttnn::Tensor> v335 = util_create_vec(v65, v140);
  ::std::vector<::ttnn::Tensor>* v336 = &g_cached_result_forward_const_eval_34;
  ttnn::constEvalFuncWrapper(v334, v335, v336);
  ::std::vector<::ttnn::Tensor> v337 = g_cached_result_forward_const_eval_34;
  ::ttnn::Tensor v338 = v337[0];
  ttnn::deallocate(v140, false);
  ttnn::deallocate(v65, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v339 = &forward_const_eval_35;
  ::std::vector<::ttnn::Tensor> v340 = util_create_vec(v99, v157);
  ::std::vector<::ttnn::Tensor>* v341 = &g_cached_result_forward_const_eval_35;
  ttnn::constEvalFuncWrapper(v339, v340, v341);
  ::std::vector<::ttnn::Tensor> v342 = g_cached_result_forward_const_eval_35;
  ::ttnn::Tensor v343 = v342[0];
  ttnn::deallocate(v157, false);
  ttnn::deallocate(v99, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v344 = &forward_const_eval_36;
  ::std::vector<::ttnn::Tensor> v345 = util_create_vec(v45, v130);
  ::std::vector<::ttnn::Tensor>* v346 = &g_cached_result_forward_const_eval_36;
  ttnn::constEvalFuncWrapper(v344, v345, v346);
  ::std::vector<::ttnn::Tensor> v347 = g_cached_result_forward_const_eval_36;
  ::ttnn::Tensor v348 = v347[0];
  ttnn::deallocate(v130, false);
  ttnn::deallocate(v45, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v349 = &forward_const_eval_37;
  ::std::vector<::ttnn::Tensor> v350 = util_create_vec(v39, v127);
  ::std::vector<::ttnn::Tensor>* v351 = &g_cached_result_forward_const_eval_37;
  ttnn::constEvalFuncWrapper(v349, v350, v351);
  ::std::vector<::ttnn::Tensor> v352 = g_cached_result_forward_const_eval_37;
  ::ttnn::Tensor v353 = v352[0];
  ttnn::deallocate(v127, false);
  ttnn::deallocate(v39, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v354 = &forward_const_eval_38;
  ::std::vector<::ttnn::Tensor> v355 = util_create_vec(v69, v142);
  ::std::vector<::ttnn::Tensor>* v356 = &g_cached_result_forward_const_eval_38;
  ttnn::constEvalFuncWrapper(v354, v355, v356);
  ::std::vector<::ttnn::Tensor> v357 = g_cached_result_forward_const_eval_38;
  ::ttnn::Tensor v358 = v357[0];
  ttnn::deallocate(v142, false);
  ttnn::deallocate(v69, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v359 = &forward_const_eval_39;
  ::std::vector<::ttnn::Tensor> v360 = util_create_vec(v73, v144);
  ::std::vector<::ttnn::Tensor>* v361 = &g_cached_result_forward_const_eval_39;
  ttnn::constEvalFuncWrapper(v359, v360, v361);
  ::std::vector<::ttnn::Tensor> v362 = g_cached_result_forward_const_eval_39;
  ::ttnn::Tensor v363 = v362[0];
  ttnn::deallocate(v144, false);
  ttnn::deallocate(v73, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v364 = &forward_const_eval_40;
  ::std::vector<::ttnn::Tensor> v365 = util_create_vec(v79, v147);
  ::std::vector<::ttnn::Tensor>* v366 = &g_cached_result_forward_const_eval_40;
  ttnn::constEvalFuncWrapper(v364, v365, v366);
  ::std::vector<::ttnn::Tensor> v367 = g_cached_result_forward_const_eval_40;
  ::ttnn::Tensor v368 = v367[0];
  ttnn::deallocate(v147, false);
  ttnn::deallocate(v79, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v369 = &forward_const_eval_41;
  ::std::vector<::ttnn::Tensor> v370 = util_create_vec(v17, v116);
  ::std::vector<::ttnn::Tensor>* v371 = &g_cached_result_forward_const_eval_41;
  ttnn::constEvalFuncWrapper(v369, v370, v371);
  ::std::vector<::ttnn::Tensor> v372 = g_cached_result_forward_const_eval_41;
  ::ttnn::Tensor v373 = v372[0];
  ttnn::deallocate(v116, false);
  ttnn::deallocate(v17, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v374 = &forward_const_eval_42;
  ::std::vector<::ttnn::Tensor> v375 = util_create_vec(v108);
  ::std::vector<::ttnn::Tensor>* v376 = &g_cached_result_forward_const_eval_42;
  ttnn::constEvalFuncWrapper(v374, v375, v376);
  ::std::vector<::ttnn::Tensor> v377 = g_cached_result_forward_const_eval_42;
  ::ttnn::Tensor v378 = v377[0];
  ttnn::deallocate(v108, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v379 = &forward_const_eval_43;
  ::std::vector<::ttnn::Tensor> v380 = util_create_vec(v155);
  ::std::vector<::ttnn::Tensor>* v381 = &g_cached_result_forward_const_eval_43;
  ttnn::constEvalFuncWrapper(v379, v380, v381);
  ::std::vector<::ttnn::Tensor> v382 = g_cached_result_forward_const_eval_43;
  ::ttnn::Tensor v383 = v382[0];
  ttnn::deallocate(v155, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v384 = &forward_const_eval_44;
  ::std::vector<::ttnn::Tensor> v385 = util_create_vec(v25, v120);
  ::std::vector<::ttnn::Tensor>* v386 = &g_cached_result_forward_const_eval_44;
  ttnn::constEvalFuncWrapper(v384, v385, v386);
  ::std::vector<::ttnn::Tensor> v387 = g_cached_result_forward_const_eval_44;
  ::ttnn::Tensor v388 = v387[0];
  ttnn::deallocate(v120, false);
  ttnn::deallocate(v25, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v389 = &forward_const_eval_45;
  ::std::vector<::ttnn::Tensor> v390 = util_create_vec(v105, v160);
  ::std::vector<::ttnn::Tensor>* v391 = &g_cached_result_forward_const_eval_45;
  ttnn::constEvalFuncWrapper(v389, v390, v391);
  ::std::vector<::ttnn::Tensor> v392 = g_cached_result_forward_const_eval_45;
  ::ttnn::Tensor v393 = v392[0];
  ttnn::deallocate(v160, false);
  ttnn::deallocate(v105, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v394 = &forward_const_eval_46;
  ::std::vector<::ttnn::Tensor> v395 = util_create_vec(v161);
  ::std::vector<::ttnn::Tensor>* v396 = &g_cached_result_forward_const_eval_46;
  ttnn::constEvalFuncWrapper(v394, v395, v396);
  ::std::vector<::ttnn::Tensor> v397 = g_cached_result_forward_const_eval_46;
  ::ttnn::Tensor v398 = v397[0];
  ttnn::deallocate(v161, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v399 = &forward_const_eval_47;
  ::std::vector<::ttnn::Tensor> v400 = util_create_vec(v5, v110);
  ::std::vector<::ttnn::Tensor>* v401 = &g_cached_result_forward_const_eval_47;
  ttnn::constEvalFuncWrapper(v399, v400, v401);
  ::std::vector<::ttnn::Tensor> v402 = g_cached_result_forward_const_eval_47;
  ::ttnn::Tensor v403 = v402[0];
  ttnn::deallocate(v110, false);
  ttnn::deallocate(v5, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v404 = &forward_const_eval_48;
  ::std::vector<::ttnn::Tensor> v405 = util_create_vec(v31, v123);
  ::std::vector<::ttnn::Tensor>* v406 = &g_cached_result_forward_const_eval_48;
  ttnn::constEvalFuncWrapper(v404, v405, v406);
  ::std::vector<::ttnn::Tensor> v407 = g_cached_result_forward_const_eval_48;
  ::ttnn::Tensor v408 = v407[0];
  ttnn::deallocate(v123, false);
  ttnn::deallocate(v31, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v409 = &forward_const_eval_49;
  ::std::vector<::ttnn::Tensor> v410 = util_create_vec(v59, v137);
  ::std::vector<::ttnn::Tensor>* v411 = &g_cached_result_forward_const_eval_49;
  ttnn::constEvalFuncWrapper(v409, v410, v411);
  ::std::vector<::ttnn::Tensor> v412 = g_cached_result_forward_const_eval_49;
  ::ttnn::Tensor v413 = v412[0];
  ttnn::deallocate(v137, false);
  ttnn::deallocate(v59, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v414 = &forward_const_eval_50;
  ::std::vector<::ttnn::Tensor> v415 = util_create_vec(v7, v111);
  ::std::vector<::ttnn::Tensor>* v416 = &g_cached_result_forward_const_eval_50;
  ttnn::constEvalFuncWrapper(v414, v415, v416);
  ::std::vector<::ttnn::Tensor> v417 = g_cached_result_forward_const_eval_50;
  ::ttnn::Tensor v418 = v417[0];
  ttnn::deallocate(v111, false);
  ttnn::deallocate(v7, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v419 = &forward_const_eval_51;
  ::std::vector<::ttnn::Tensor> v420 = util_create_vec(v27, v121);
  ::std::vector<::ttnn::Tensor>* v421 = &g_cached_result_forward_const_eval_51;
  ttnn::constEvalFuncWrapper(v419, v420, v421);
  ::std::vector<::ttnn::Tensor> v422 = g_cached_result_forward_const_eval_51;
  ::ttnn::Tensor v423 = v422[0];
  ttnn::deallocate(v121, false);
  ttnn::deallocate(v27, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v424 = &forward_const_eval_52;
  ::std::vector<::ttnn::Tensor> v425 = util_create_vec(v35, v125);
  ::std::vector<::ttnn::Tensor>* v426 = &g_cached_result_forward_const_eval_52;
  ttnn::constEvalFuncWrapper(v424, v425, v426);
  ::std::vector<::ttnn::Tensor> v427 = g_cached_result_forward_const_eval_52;
  ::ttnn::Tensor v428 = v427[0];
  ttnn::deallocate(v125, false);
  ttnn::deallocate(v35, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v429 = &forward_const_eval_53;
  ::std::vector<::ttnn::Tensor> v430 = util_create_vec(v47, v131);
  ::std::vector<::ttnn::Tensor>* v431 = &g_cached_result_forward_const_eval_53;
  ttnn::constEvalFuncWrapper(v429, v430, v431);
  ::std::vector<::ttnn::Tensor> v432 = g_cached_result_forward_const_eval_53;
  ::ttnn::Tensor v433 = v432[0];
  ttnn::deallocate(v131, false);
  ttnn::deallocate(v47, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v434 = &forward_const_eval_54;
  ::std::vector<::ttnn::Tensor> v435 = util_create_vec(v71, v143);
  ::std::vector<::ttnn::Tensor>* v436 = &g_cached_result_forward_const_eval_54;
  ttnn::constEvalFuncWrapper(v434, v435, v436);
  ::std::vector<::ttnn::Tensor> v437 = g_cached_result_forward_const_eval_54;
  ::ttnn::Tensor v438 = v437[0];
  ttnn::deallocate(v143, false);
  ttnn::deallocate(v71, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v439 = &forward_const_eval_55;
  ::std::vector<::ttnn::Tensor> v440 = util_create_vec(v89, v152);
  ::std::vector<::ttnn::Tensor>* v441 = &g_cached_result_forward_const_eval_55;
  ttnn::constEvalFuncWrapper(v439, v440, v441);
  ::std::vector<::ttnn::Tensor> v442 = g_cached_result_forward_const_eval_55;
  ::ttnn::Tensor v443 = v442[0];
  ttnn::deallocate(v152, false);
  ttnn::deallocate(v89, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v444 = &forward_const_eval_56;
  ::std::vector<::ttnn::Tensor> v445 = util_create_vec(v77, v146);
  ::std::vector<::ttnn::Tensor>* v446 = &g_cached_result_forward_const_eval_56;
  ttnn::constEvalFuncWrapper(v444, v445, v446);
  ::std::vector<::ttnn::Tensor> v447 = g_cached_result_forward_const_eval_56;
  ::ttnn::Tensor v448 = v447[0];
  ttnn::deallocate(v146, false);
  ttnn::deallocate(v77, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v449 = &forward_const_eval_57;
  ::std::vector<::ttnn::Tensor> v450 = util_create_vec(v29, v122);
  ::std::vector<::ttnn::Tensor>* v451 = &g_cached_result_forward_const_eval_57;
  ttnn::constEvalFuncWrapper(v449, v450, v451);
  ::std::vector<::ttnn::Tensor> v452 = g_cached_result_forward_const_eval_57;
  ::ttnn::Tensor v453 = v452[0];
  ttnn::deallocate(v122, false);
  ttnn::deallocate(v29, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v454 = &forward_const_eval_58;
  ::std::vector<::ttnn::Tensor> v455 = util_create_vec(v95);
  ::std::vector<::ttnn::Tensor>* v456 = &g_cached_result_forward_const_eval_58;
  ttnn::constEvalFuncWrapper(v454, v455, v456);
  ::std::vector<::ttnn::Tensor> v457 = g_cached_result_forward_const_eval_58;
  ::ttnn::Tensor v458 = v457[0];
  ttnn::deallocate(v95, false);
  ttnn::distributed::MeshDevice* v459 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v460 = ttnn::transpose(v2, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v461 = ttnn::transpose(v460, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v460, false);
  ::ttnn::Tensor v462 = ttnn::reshape(v461, ::std::vector<int32_t>{1, 1, 401408, 3}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v461, false);
  ::ttnn::Tensor v463 = ::std::get<0>(ttnn::conv2d(v462, v238, v459, 3, 64, 8, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{3, 3, 3, 3}, ::std::array<uint32_t, 2>{1, 1}, 1, v4, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v462, false);
  ttnn::deallocate(v238, false);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v464 = ttnn::max_pool2d(v463, 8, 112, 112, 64, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::std::nullopt, false);
  ttnn::deallocate(v463, false);
  ::ttnn::Tensor v465 = ttnn::to_layout(v464, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v459);
  ::ttnn::Tensor v466 = ttnn::to_memory_config(v465, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v465, false);
  ::ttnn::Tensor v467 = ::std::get<0>(ttnn::conv2d(v466, v208, v459, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v12, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v466, false);
  ttnn::deallocate(v208, false);
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v468 = ttnn::to_memory_config(v467, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v467, false);
  ::ttnn::Tensor v469 = ttnn::to_layout(v464, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, v459);
  ttnn::deallocate(v464, false);
  ::ttnn::Tensor v470 = ttnn::to_memory_config(v469, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v469, false);
  ::ttnn::Tensor v471 = ::std::get<0>(ttnn::conv2d(v470, v403, v459, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v6, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v470, false);
  ttnn::deallocate(v403, false);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v472 = ::std::get<0>(ttnn::conv2d(v471, v418, v459, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v8, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v471, false);
  ttnn::deallocate(v418, false);
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v473 = ::std::get<0>(ttnn::conv2d(v472, v273, v459, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v10, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v472, false);
  ttnn::deallocate(v273, false);
  ttnn::deallocate(v10, false);
  ::ttnn::Tensor v474 = ttnn::add(v473, v468, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v473, false);
  ttnn::deallocate(v468, false);
  ::ttnn::Tensor v475 = ttnn::relu(v474, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v474, false);
  ::ttnn::Tensor v476 = ttnn::to_memory_config(v475, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v477 = ::std::get<0>(ttnn::conv2d(v475, v318, v459, 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v14, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v475, false);
  ttnn::deallocate(v318, false);
  ttnn::deallocate(v14, false);
  ::ttnn::Tensor v478 = ::std::get<0>(ttnn::conv2d(v477, v193, v459, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v16, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v477, false);
  ttnn::deallocate(v193, false);
  ttnn::deallocate(v16, false);
  ::ttnn::Tensor v479 = ::std::get<0>(ttnn::conv2d(v478, v373, v459, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v18, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v478, false);
  ttnn::deallocate(v373, false);
  ttnn::deallocate(v18, false);
  ::ttnn::Tensor v480 = ttnn::add(v479, v476, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v479, false);
  ttnn::deallocate(v476, false);
  ::ttnn::Tensor v481 = ttnn::relu(v480, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v480, false);
  ::ttnn::Tensor v482 = ttnn::to_memory_config(v481, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v483 = ::std::get<0>(ttnn::conv2d(v481, v278, v459, 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v20, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v481, false);
  ttnn::deallocate(v278, false);
  ttnn::deallocate(v20, false);
  ::ttnn::Tensor v484 = ::std::get<0>(ttnn::conv2d(v483, v258, v459, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v22, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v483, false);
  ttnn::deallocate(v258, false);
  ttnn::deallocate(v22, false);
  ::ttnn::Tensor v485 = ::std::get<0>(ttnn::conv2d(v484, v253, v459, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v24, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v484, false);
  ttnn::deallocate(v253, false);
  ttnn::deallocate(v24, false);
  ::ttnn::Tensor v486 = ttnn::add(v485, v482, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v485, false);
  ttnn::deallocate(v482, false);
  ::ttnn::Tensor v487 = ttnn::relu(v486, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v486, false);
  ::ttnn::Tensor v488 = ttnn::to_memory_config(v487, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v489 = ttnn::to_memory_config(v488, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{3136, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v488, false);
  ::ttnn::Tensor v490 = ::std::get<0>(ttnn::conv2d(v489, v408, v459, 256, 512, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v32, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v489, false);
  ttnn::deallocate(v408, false);
  ttnn::deallocate(v32, false);
  ::ttnn::Tensor v491 = ttnn::to_memory_config(v490, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v490, false);
  ::ttnn::Tensor v492 = ::std::get<0>(ttnn::conv2d(v487, v388, v459, 256, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v26, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v487, false);
  ttnn::deallocate(v388, false);
  ttnn::deallocate(v26, false);
  ::ttnn::Tensor v493 = ttnn::to_memory_config(v492, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{512, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v492, false);
  ::ttnn::Tensor v494 = ::std::get<0>(ttnn::conv2d(v493, v423, v459, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v28, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v493, false);
  ttnn::deallocate(v423, false);
  ttnn::deallocate(v28, false);
  ::ttnn::Tensor v495 = ::std::get<0>(ttnn::conv2d(v494, v453, v459, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v30, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v494, false);
  ttnn::deallocate(v453, false);
  ttnn::deallocate(v30, false);
  ::ttnn::Tensor v496 = ttnn::add(v495, v491, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v495, false);
  ttnn::deallocate(v491, false);
  ::ttnn::Tensor v497 = ttnn::relu(v496, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v496, false);
  ::ttnn::Tensor v498 = ttnn::to_memory_config(v497, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v499 = ::std::get<0>(ttnn::conv2d(v497, v213, v459, 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v34, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v497, false);
  ttnn::deallocate(v213, false);
  ttnn::deallocate(v34, false);
  ::ttnn::Tensor v500 = ::std::get<0>(ttnn::conv2d(v499, v428, v459, 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v36, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v499, false);
  ttnn::deallocate(v428, false);
  ttnn::deallocate(v36, false);
  ::ttnn::Tensor v501 = ::std::get<0>(ttnn::conv2d(v500, v178, v459, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v38, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v500, false);
  ttnn::deallocate(v178, false);
  ttnn::deallocate(v38, false);
  ::ttnn::Tensor v502 = ttnn::add(v501, v498, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v501, false);
  ttnn::deallocate(v498, false);
  ::ttnn::Tensor v503 = ttnn::relu(v502, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v502, false);
  ::ttnn::Tensor v504 = ttnn::to_memory_config(v503, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v505 = ::std::get<0>(ttnn::conv2d(v503, v353, v459, 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v40, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v503, false);
  ttnn::deallocate(v353, false);
  ttnn::deallocate(v40, false);
  ::ttnn::Tensor v506 = ::std::get<0>(ttnn::conv2d(v505, v263, v459, 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v42, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v505, false);
  ttnn::deallocate(v263, false);
  ttnn::deallocate(v42, false);
  ::ttnn::Tensor v507 = ::std::get<0>(ttnn::conv2d(v506, v218, v459, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v44, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v506, false);
  ttnn::deallocate(v218, false);
  ttnn::deallocate(v44, false);
  ::ttnn::Tensor v508 = ttnn::add(v507, v504, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v507, false);
  ttnn::deallocate(v504, false);
  ::ttnn::Tensor v509 = ttnn::relu(v508, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v508, false);
  ::ttnn::Tensor v510 = ttnn::to_memory_config(v509, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v511 = ::std::get<0>(ttnn::conv2d(v509, v348, v459, 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v46, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v509, false);
  ttnn::deallocate(v348, false);
  ttnn::deallocate(v46, false);
  ::ttnn::Tensor v512 = ::std::get<0>(ttnn::conv2d(v511, v433, v459, 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v48, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v511, false);
  ttnn::deallocate(v433, false);
  ttnn::deallocate(v48, false);
  ::ttnn::Tensor v513 = ::std::get<0>(ttnn::conv2d(v512, v173, v459, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v50, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v512, false);
  ttnn::deallocate(v173, false);
  ttnn::deallocate(v50, false);
  ::ttnn::Tensor v514 = ttnn::add(v513, v510, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v513, false);
  ttnn::deallocate(v510, false);
  ::ttnn::Tensor v515 = ttnn::relu(v514, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v514, false);
  ::ttnn::Tensor v516 = ttnn::to_memory_config(v515, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v517 = ttnn::to_memory_config(v516, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v516, false);
  ::ttnn::Tensor v518 = ::std::get<0>(ttnn::conv2d(v517, v298, v459, 512, 1024, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v58, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v517, false);
  ttnn::deallocate(v298, false);
  ttnn::deallocate(v58, false);
  ::ttnn::Tensor v519 = ttnn::to_memory_config(v518, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v518, false);
  ::ttnn::Tensor v520 = ttnn::to_memory_config(v515, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v515, false);
  ::ttnn::Tensor v521 = ::std::get<0>(ttnn::conv2d(v520, v268, v459, 512, 256, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v52, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v520, false);
  ttnn::deallocate(v268, false);
  ttnn::deallocate(v52, false);
  ::ttnn::Tensor v522 = ttnn::to_memory_config(v521, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v521, false);
  ::ttnn::Tensor v523 = ::std::get<0>(ttnn::conv2d(v522, v323, v459, 256, 256, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v54, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v522, false);
  ttnn::deallocate(v323, false);
  ttnn::deallocate(v54, false);
  ::ttnn::Tensor v524 = ::std::get<0>(ttnn::conv2d(v523, v248, v459, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v56, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v523, false);
  ttnn::deallocate(v248, false);
  ttnn::deallocate(v56, false);
  ::ttnn::Tensor v525 = ttnn::add(v524, v519, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v524, false);
  ttnn::deallocate(v519, false);
  ::ttnn::Tensor v526 = ttnn::relu(v525, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v525, false);
  ::ttnn::Tensor v527 = ttnn::to_memory_config(v526, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v528 = ::std::get<0>(ttnn::conv2d(v526, v413, v459, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v60, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v526, false);
  ttnn::deallocate(v413, false);
  ttnn::deallocate(v60, false);
  ::ttnn::Tensor v529 = ::std::get<0>(ttnn::conv2d(v528, v283, v459, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v62, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v528, false);
  ttnn::deallocate(v283, false);
  ttnn::deallocate(v62, false);
  ::ttnn::Tensor v530 = ::std::get<0>(ttnn::conv2d(v529, v223, v459, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v64, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v529, false);
  ttnn::deallocate(v223, false);
  ttnn::deallocate(v64, false);
  ::ttnn::Tensor v531 = ttnn::add(v530, v527, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v530, false);
  ttnn::deallocate(v527, false);
  ::ttnn::Tensor v532 = ttnn::relu(v531, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v531, false);
  ::ttnn::Tensor v533 = ttnn::to_memory_config(v532, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v534 = ::std::get<0>(ttnn::conv2d(v532, v338, v459, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v66, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v532, false);
  ttnn::deallocate(v338, false);
  ttnn::deallocate(v66, false);
  ::ttnn::Tensor v535 = ::std::get<0>(ttnn::conv2d(v534, v188, v459, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v68, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v534, false);
  ttnn::deallocate(v188, false);
  ttnn::deallocate(v68, false);
  ::ttnn::Tensor v536 = ::std::get<0>(ttnn::conv2d(v535, v358, v459, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v70, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v535, false);
  ttnn::deallocate(v358, false);
  ttnn::deallocate(v70, false);
  ::ttnn::Tensor v537 = ttnn::add(v536, v533, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v536, false);
  ttnn::deallocate(v533, false);
  ::ttnn::Tensor v538 = ttnn::relu(v537, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v537, false);
  ::ttnn::Tensor v539 = ttnn::to_memory_config(v538, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v540 = ::std::get<0>(ttnn::conv2d(v538, v438, v459, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v72, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v538, false);
  ttnn::deallocate(v438, false);
  ttnn::deallocate(v72, false);
  ::ttnn::Tensor v541 = ::std::get<0>(ttnn::conv2d(v540, v363, v459, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v74, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v540, false);
  ttnn::deallocate(v363, false);
  ttnn::deallocate(v74, false);
  ::ttnn::Tensor v542 = ::std::get<0>(ttnn::conv2d(v541, v233, v459, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v76, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v541, false);
  ttnn::deallocate(v233, false);
  ttnn::deallocate(v76, false);
  ::ttnn::Tensor v543 = ttnn::add(v542, v539, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v542, false);
  ttnn::deallocate(v539, false);
  ::ttnn::Tensor v544 = ttnn::relu(v543, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v543, false);
  ::ttnn::Tensor v545 = ttnn::to_memory_config(v544, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v546 = ::std::get<0>(ttnn::conv2d(v544, v448, v459, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v78, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v544, false);
  ttnn::deallocate(v448, false);
  ttnn::deallocate(v78, false);
  ::ttnn::Tensor v547 = ::std::get<0>(ttnn::conv2d(v546, v368, v459, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v80, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v546, false);
  ttnn::deallocate(v368, false);
  ttnn::deallocate(v80, false);
  ::ttnn::Tensor v548 = ::std::get<0>(ttnn::conv2d(v547, v313, v459, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v82, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v547, false);
  ttnn::deallocate(v313, false);
  ttnn::deallocate(v82, false);
  ::ttnn::Tensor v549 = ttnn::add(v548, v545, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v548, false);
  ttnn::deallocate(v545, false);
  ::ttnn::Tensor v550 = ttnn::relu(v549, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v549, false);
  ::ttnn::Tensor v551 = ttnn::to_memory_config(v550, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v552 = ::std::get<0>(ttnn::conv2d(v550, v243, v459, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v84, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v550, false);
  ttnn::deallocate(v243, false);
  ttnn::deallocate(v84, false);
  ::ttnn::Tensor v553 = ::std::get<0>(ttnn::conv2d(v552, v293, v459, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v86, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v552, false);
  ttnn::deallocate(v293, false);
  ttnn::deallocate(v86, false);
  ::ttnn::Tensor v554 = ::std::get<0>(ttnn::conv2d(v553, v168, v459, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v88, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v553, false);
  ttnn::deallocate(v168, false);
  ttnn::deallocate(v88, false);
  ::ttnn::Tensor v555 = ttnn::add(v554, v551, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v554, false);
  ttnn::deallocate(v551, false);
  ::ttnn::Tensor v556 = ttnn::relu(v555, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v555, false);
  ::ttnn::Tensor v557 = ttnn::to_memory_config(v556, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v558 = ::std::get<0>(ttnn::conv2d(v556, v443, v459, 1024, 512, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v90, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v556, false);
  ttnn::deallocate(v443, false);
  ttnn::deallocate(v90, false);
  ::ttnn::Tensor v559 = ::std::get<0>(ttnn::conv2d(v558, v228, v459, 512, 512, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v92, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v558, false);
  ttnn::deallocate(v228, false);
  ttnn::deallocate(v92, false);
  ::ttnn::Tensor v560 = ::std::get<0>(ttnn::conv2d(v559, v198, v459, 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v559, false);
  ttnn::deallocate(v198, false);
  ::ttnn::Tensor v561 = ttnn::reshape(v560, ::std::vector<int32_t>{8, 7, 7, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v560, false);
  ::ttnn::Tensor v562 = ttnn::transpose(v561, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v561, false);
  ::ttnn::Tensor v563 = ttnn::transpose(v562, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v562, false);
  ::ttnn::Tensor v564 = ttnn::multiply(v563, v203, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{8192, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v563, false);
  ttnn::deallocate(v203, false);
  ::ttnn::Tensor v565 = ttnn::add(v564, v94, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{8192, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v564, false);
  ttnn::deallocate(v94, false);
  ::ttnn::Tensor v566 = ttnn::to_memory_config(v565, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v565, false);
  ::ttnn::Tensor v567 = ::std::get<0>(ttnn::conv2d(v557, v383, v459, 1024, 2048, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v557, false);
  ttnn::deallocate(v383, false);
  ::ttnn::Tensor v568 = ttnn::reshape(v567, ::std::vector<int32_t>{8, 7, 7, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v567, false);
  ::ttnn::Tensor v569 = ttnn::transpose(v568, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v568, false);
  ::ttnn::Tensor v570 = ttnn::transpose(v569, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v569, false);
  ::ttnn::Tensor v571 = ttnn::multiply(v570, v458, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v570, false);
  ttnn::deallocate(v458, false);
  ::ttnn::Tensor v572 = ttnn::add(v571, v96, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v571, false);
  ttnn::deallocate(v96, false);
  ::ttnn::Tensor v573 = ttnn::add(v566, v572, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v572, false);
  ttnn::deallocate(v566, false);
  ::ttnn::Tensor v574 = ttnn::relu(v573, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v573, false);
  ::ttnn::Tensor v575 = ttnn::transpose(v574, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v576 = ttnn::transpose(v575, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v575, false);
  ::ttnn::Tensor v577 = ttnn::reshape(v576, ::std::vector<int32_t>{1, 1, 392, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v576, false);
  ::ttnn::Tensor v578 = ttnn::to_memory_config(v577, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v577, false);
  ::ttnn::Tensor v579 = ::std::get<0>(ttnn::conv2d(v578, v333, v459, 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v98, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v578, false);
  ttnn::deallocate(v333, false);
  ttnn::deallocate(v98, false);
  ::ttnn::Tensor v580 = ::std::get<0>(ttnn::conv2d(v579, v343, v459, 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v100, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v579, false);
  ttnn::deallocate(v343, false);
  ttnn::deallocate(v100, false);
  ::ttnn::Tensor v581 = ::std::get<0>(ttnn::conv2d(v580, v308, v459, 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v580, false);
  ttnn::deallocate(v308, false);
  ::ttnn::Tensor v582 = ttnn::reshape(v581, ::std::vector<int32_t>{8, 7, 7, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v581, false);
  ::ttnn::Tensor v583 = ttnn::transpose(v582, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v582, false);
  ::ttnn::Tensor v584 = ttnn::transpose(v583, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v583, false);
  ::ttnn::Tensor v585 = ttnn::multiply(v584, v183, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{8192, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v584, false);
  ttnn::deallocate(v183, false);
  ::ttnn::Tensor v586 = ttnn::add(v585, v102, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{8192, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v585, false);
  ttnn::deallocate(v102, false);
  ::ttnn::Tensor v587 = ttnn::add(v586, v574, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{8192, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v586, false);
  ttnn::deallocate(v574, false);
  ::ttnn::Tensor v588 = ttnn::relu(v587, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{8192, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v587, false);
  ::ttnn::Tensor v589 = ttnn::to_memory_config(v588, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v588, false);
  ::ttnn::Tensor v590 = ttnn::transpose(v589, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v591 = ttnn::transpose(v590, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v590, false);
  ::ttnn::Tensor v592 = ttnn::reshape(v591, ::std::vector<int32_t>{1, 1, 392, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v591, false);
  ::ttnn::Tensor v593 = ttnn::to_memory_config(v592, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v592, false);
  ::ttnn::Tensor v594 = ::std::get<0>(ttnn::conv2d(v593, v328, v459, 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v104, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v593, false);
  ttnn::deallocate(v328, false);
  ttnn::deallocate(v104, false);
  ::ttnn::Tensor v595 = ::std::get<0>(ttnn::conv2d(v594, v393, v459, 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v106, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v594, false);
  ttnn::deallocate(v393, false);
  ttnn::deallocate(v106, false);
  ::ttnn::Tensor v596 = ::std::get<0>(ttnn::conv2d(v595, v398, v459, 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v595, false);
  ttnn::deallocate(v398, false);
  ::ttnn::Tensor v597 = ttnn::reshape(v596, ::std::vector<int32_t>{8, 7, 7, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v596, false);
  ::ttnn::Tensor v598 = ttnn::transpose(v597, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v597, false);
  ::ttnn::Tensor v599 = ttnn::transpose(v598, 1, 2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v598, false);
  ::ttnn::Tensor v600 = ttnn::reshape(v599, ::std::vector<int32_t>{8, 1, 2048, 49}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v599, false);
  ::ttnn::Tensor v601 = ttnn::transpose(v600, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v600, false);
  ::ttnn::Tensor v602 = ttnn::multiply(v601, v303, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{512, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v601, false);
  ttnn::deallocate(v303, false);
  ::ttnn::Tensor v603 = ttnn::add(v602, v378, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{512, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v602, false);
  ttnn::deallocate(v378, false);
  ::ttnn::Tensor v604 = ttnn::to_memory_config(v603, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v604, false);
  ::ttnn::Tensor v605 = ttnn::reshape(v589, ::std::vector<int32_t>{8, 1, 2048, 49}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v589, false);
  ::ttnn::Tensor v606 = ttnn::transpose(v605, 2, 3, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v605, false);
  ::ttnn::Tensor v607 = ttnn::add(v603, v606, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{512, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v606, false);
  ttnn::deallocate(v603, false);
  ::ttnn::Tensor v608 = ttnn::relu(v607, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::WIDTH_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{512, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v607, false);
  ::ttnn::Tensor v609 = ttnn::to_memory_config(v608, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v608, false);
  ::ttnn::Tensor v610 = ttnn::mean(v609, ::ttnn::SmallVector<int32_t>{-2}, true, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v609, false);
  ::ttnn::Tensor v611 = ttnn::reshape(v610, ::std::vector<int32_t>{8, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v610, false);
  ::ttnn::Tensor v612 = ttnn::matmul(v611, v162, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v611, false);
  ttnn::deallocate(v162, false);
  ::ttnn::Tensor v613 = ttnn::add(v612, v288, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v612, false);
  ttnn::deallocate(v288, false);
  ::std::vector<::ttnn::Tensor> v614 = util_create_vec(v613);
  return v614;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_0() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_1() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_2() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_3() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v4 = util_create_vec(v3);
  return v4;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_4() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_5() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_6() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_7() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v4 = util_create_vec(v3);
  return v4;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_8() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_9() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_10() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_11() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_12() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_13() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_14() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 3, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_15() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_16() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_17() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_18() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_19() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_20() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_21() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_22() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_23() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_24() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1000}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v4 = util_create_vec(v3);
  return v4;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_25() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_26() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_27() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v4 = util_create_vec(v3);
  return v4;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_28() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_29() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_30() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_31() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_32() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_33() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_34() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_35() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_36() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_37() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_38() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_39() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_40() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_41() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_42() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v4 = util_create_vec(v3);
  return v4;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_43() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_44() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_45() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_46() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_47() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_48() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_49() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_50() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_51() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_52() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_53() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_54() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_55() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_56() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_57() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v3, v4);
  return v5;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_58() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v4 = util_create_vec(v3);
  return v4;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({8, 3, 224, 224}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v3 = v2;
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_device(v6, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::to_device(v8, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::to_device(v10, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::to_device(v12, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::to_device(v14, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::to_device(v16, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::to_device(v18, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::to_device(v20, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::to_device(v22, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::to_device(v24, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::to_device(v26, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::to_device(v28, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::to_device(v30, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::to_device(v32, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::to_device(v34, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::to_device(v36, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::to_device(v38, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::to_device(v40, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::to_device(v42, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::to_device(v44, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::to_device(v46, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::to_device(v48, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::to_device(v50, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::to_device(v52, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::to_device(v54, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v57 = ttnn::to_device(v56, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v58 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v59 = ttnn::to_device(v58, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v60 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v61 = ttnn::to_device(v60, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v62 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v63 = ttnn::to_device(v62, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v64 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v65 = ttnn::to_device(v64, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v66 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v67 = ttnn::to_device(v66, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v68 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v69 = ttnn::to_device(v68, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v70 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v71 = ttnn::to_device(v70, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v72 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v73 = ttnn::to_device(v72, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v74 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v75 = ttnn::to_device(v74, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v76 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v77 = ttnn::to_device(v76, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v78 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v79 = ttnn::to_device(v78, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v80 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v81 = ttnn::to_device(v80, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v82 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v83 = ttnn::to_device(v82, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v84 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v85 = ttnn::to_device(v84, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v86 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v87 = ttnn::to_device(v86, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v88 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v89 = ttnn::to_device(v88, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v90 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v91 = ttnn::to_device(v90, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v92 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v93 = ttnn::to_device(v92, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v94 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v95 = ttnn::to_device(v94, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v96 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v97 = ttnn::to_device(v96, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v98 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v99 = ttnn::to_device(v98, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v100 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v101 = ttnn::to_device(v100, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v102 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v103 = ttnn::to_device(v102, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v104 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v105 = ttnn::to_device(v104, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v106 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v107 = ttnn::to_device(v106, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v108 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v109 = ttnn::to_device(v108, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v110 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v111 = ttnn::to_device(v110, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v112 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v113 = ttnn::to_device(v112, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v114 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v115 = ttnn::to_device(v114, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v116 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v117 = ttnn::to_device(v116, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v118 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v119 = ttnn::to_device(v118, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v120 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v121 = ttnn::to_device(v120, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v122 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v123 = ttnn::to_device(v122, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v124 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v125 = ttnn::to_device(v124, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v126 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v127 = ttnn::to_device(v126, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v128 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v129 = ttnn::to_device(v128, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v130 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v131 = ttnn::to_device(v130, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v132 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v133 = ttnn::to_device(v132, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v134 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v135 = ttnn::to_device(v134, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v136 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v137 = ttnn::to_device(v136, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v138 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v139 = ttnn::to_device(v138, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v140 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v141 = ttnn::to_device(v140, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v142 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v143 = ttnn::to_device(v142, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v144 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v145 = ttnn::to_device(v144, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v146 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v147 = ttnn::to_device(v146, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v148 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v149 = ttnn::to_device(v148, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v150 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v151 = ttnn::to_device(v150, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v152 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v153 = ttnn::to_device(v152, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v154 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v155 = ttnn::to_device(v154, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v156 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v157 = ttnn::to_device(v156, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v158 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v159 = ttnn::to_device(v158, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v160 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v161 = ttnn::to_device(v160, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v162 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v163 = ttnn::to_device(v162, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v164 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v165 = ttnn::to_device(v164, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v166 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v167 = ttnn::to_device(v166, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v168 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v169 = ttnn::to_device(v168, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v170 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v171 = ttnn::to_device(v170, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v172 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v173 = ttnn::to_device(v172, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v174 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v175 = ttnn::to_device(v174, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v176 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v177 = ttnn::to_device(v176, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v178 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v179 = ttnn::to_device(v178, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v180 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v181 = ttnn::to_device(v180, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v182 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v183 = ttnn::to_device(v182, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v184 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v185 = ttnn::to_device(v184, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v186 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v187 = ttnn::to_device(v186, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v188 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v189 = ttnn::to_device(v188, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v190 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v191 = ttnn::to_device(v190, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v192 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v193 = ttnn::to_device(v192, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v194 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v195 = ttnn::to_device(v194, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v196 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v197 = ttnn::to_device(v196, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v198 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v199 = ttnn::to_device(v198, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v200 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v201 = ttnn::to_device(v200, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v202 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v203 = ttnn::to_device(v202, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v204 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v205 = ttnn::to_device(v204, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v206 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v207 = ttnn::to_device(v206, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v208 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v209 = ttnn::to_device(v208, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v210 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v211 = ttnn::to_device(v210, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v212 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v213 = ttnn::to_device(v212, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v214 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v215 = ttnn::to_device(v214, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v216 = ttnn::ones(::ttnn::Shape({64, 3, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v217 = ttnn::ones(::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v218 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v219 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v220 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v221 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v222 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v223 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v224 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v225 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v226 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v227 = ttnn::ones(::ttnn::Shape({128, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v228 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v229 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v230 = ttnn::ones(::ttnn::Shape({512, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v231 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v232 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v233 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v234 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v235 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v236 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v237 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v238 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v239 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v240 = ttnn::ones(::ttnn::Shape({256, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v241 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v242 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v243 = ttnn::ones(::ttnn::Shape({1024, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v244 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v245 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v246 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v247 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v248 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v249 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v250 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v251 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v252 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v253 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v254 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v255 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v256 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v257 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v258 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v259 = ttnn::ones(::ttnn::Shape({512, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v260 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v261 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v262 = ttnn::ones(::ttnn::Shape({2048, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v263 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v264 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v265 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v266 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v267 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v268 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v269 = ttnn::ones(::ttnn::Shape({2048, 1000}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v270 = ttnn::to_device(v269, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v271 = ttnn::ones(::ttnn::Shape({1000}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v272 = ttnn::to_device(v271, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v273 = util_create_vec(v3, v5, v7, v9, v11, v13, v15, v17, v19, v21, v23, v25, v27, v29, v31, v33, v35, v37, v39, v41, v43, v45, v47, v49, v51, v53, v55, v57, v59, v61, v63, v65, v67, v69, v71, v73, v75, v77, v79, v81, v83, v85, v87, v89, v91, v93, v95, v97, v99, v101, v103, v105, v107, v109, v111, v113, v115, v117, v119, v121, v123, v125, v127, v129, v131, v133, v135, v137, v139, v141, v143, v145, v147, v149, v151, v153, v155, v157, v159, v161, v163, v165, v167, v169, v171, v173, v175, v177, v179, v181, v183, v185, v187, v189, v191, v193, v195, v197, v199, v201, v203, v205, v207, v209, v211, v213, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241, v242, v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255, v256, v257, v258, v259, v260, v261, v262, v263, v264, v265, v266, v267, v268, v270, v272);
  return v273;
}

ttnn::Tensor host_input_tensor() {
  return ttnn::ones(::ttnn::Shape({8, 3, 224, 224}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
}

std::pair<ttnn::Tensor, ttnn::MeshTraceId> capture_trace(const std::vector<ttnn::Tensor> &inputs, ttnn::MeshDevice *device) {
  auto tid = ttnn::operations::trace::begin_trace_capture(device, ttnn::DefaultQueueId);
  auto output = forward(inputs);
  ttnn::operations::trace::end_trace_capture(device, tid, ttnn::DefaultQueueId);
  return std::make_pair(output[0], tid);
}

void execute_trace(ttnn::MeshDevice *device, const ttnn::MeshTraceId &tid) {
  ttnn::operations::trace::execute_trace(device, tid, ttnn::DefaultQueueId, /*blocking=*/true);
}

double time_run_wo_trace(const std::string &run_name, std::vector<::ttnn::Tensor> &v, const ttnn::Tensor &host_input, ttnn::MeshDevice *device) {
  auto start = std::chrono::high_resolution_clock::now();
  v[0] = ttnn::to_device(host_input, device, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = forward(v);
  auto t = ttnn::from_device(v2[0]);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // Print results
  //
  std::cout << duration.count() << " seconds for run: " << run_name << std::endl;

  return duration.count();
}

double time_run_with_trace(ttnn::MeshDevice *device, const ttnn::MeshTraceId &tid, const ttnn::Tensor &host_input, ttnn::Tensor &device_input) {
  auto start = std::chrono::high_resolution_clock::now();
  tt::tt_metal::write_tensor(host_input, device_input);
  execute_trace(device, tid);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // Print results
  //
  std::cout << duration.count() << " seconds for run: execute/w-trace" << std::endl;

  return duration.count();
}

int32_t main() {
  ::std::vector<::ttnn::Tensor> v1 = create_inputs_for_forward_const_eval_0();
  ::std::vector<::ttnn::Tensor> v2 = forward_const_eval_0(v1);
  ::std::vector<::ttnn::Tensor> v3 = create_inputs_for_forward_const_eval_1();
  ::std::vector<::ttnn::Tensor> v4 = forward_const_eval_1(v3);
  ::std::vector<::ttnn::Tensor> v5 = create_inputs_for_forward_const_eval_2();
  ::std::vector<::ttnn::Tensor> v6 = forward_const_eval_2(v5);
  ::std::vector<::ttnn::Tensor> v7 = create_inputs_for_forward_const_eval_3();
  ::std::vector<::ttnn::Tensor> v8 = forward_const_eval_3(v7);
  ::std::vector<::ttnn::Tensor> v9 = create_inputs_for_forward_const_eval_4();
  ::std::vector<::ttnn::Tensor> v10 = forward_const_eval_4(v9);
  ::std::vector<::ttnn::Tensor> v11 = create_inputs_for_forward_const_eval_5();
  ::std::vector<::ttnn::Tensor> v12 = forward_const_eval_5(v11);
  ::std::vector<::ttnn::Tensor> v13 = create_inputs_for_forward_const_eval_6();
  ::std::vector<::ttnn::Tensor> v14 = forward_const_eval_6(v13);
  ::std::vector<::ttnn::Tensor> v15 = create_inputs_for_forward_const_eval_7();
  ::std::vector<::ttnn::Tensor> v16 = forward_const_eval_7(v15);
  ::std::vector<::ttnn::Tensor> v17 = create_inputs_for_forward_const_eval_8();
  ::std::vector<::ttnn::Tensor> v18 = forward_const_eval_8(v17);
  ::std::vector<::ttnn::Tensor> v19 = create_inputs_for_forward_const_eval_9();
  ::std::vector<::ttnn::Tensor> v20 = forward_const_eval_9(v19);
  ::std::vector<::ttnn::Tensor> v21 = create_inputs_for_forward_const_eval_10();
  ::std::vector<::ttnn::Tensor> v22 = forward_const_eval_10(v21);
  ::std::vector<::ttnn::Tensor> v23 = create_inputs_for_forward_const_eval_11();
  ::std::vector<::ttnn::Tensor> v24 = forward_const_eval_11(v23);
  ::std::vector<::ttnn::Tensor> v25 = create_inputs_for_forward_const_eval_12();
  ::std::vector<::ttnn::Tensor> v26 = forward_const_eval_12(v25);
  ::std::vector<::ttnn::Tensor> v27 = create_inputs_for_forward_const_eval_13();
  ::std::vector<::ttnn::Tensor> v28 = forward_const_eval_13(v27);
  ::std::vector<::ttnn::Tensor> v29 = create_inputs_for_forward_const_eval_14();
  ::std::vector<::ttnn::Tensor> v30 = forward_const_eval_14(v29);
  ::std::vector<::ttnn::Tensor> v31 = create_inputs_for_forward_const_eval_15();
  ::std::vector<::ttnn::Tensor> v32 = forward_const_eval_15(v31);
  ::std::vector<::ttnn::Tensor> v33 = create_inputs_for_forward_const_eval_16();
  ::std::vector<::ttnn::Tensor> v34 = forward_const_eval_16(v33);
  ::std::vector<::ttnn::Tensor> v35 = create_inputs_for_forward_const_eval_17();
  ::std::vector<::ttnn::Tensor> v36 = forward_const_eval_17(v35);
  ::std::vector<::ttnn::Tensor> v37 = create_inputs_for_forward_const_eval_18();
  ::std::vector<::ttnn::Tensor> v38 = forward_const_eval_18(v37);
  ::std::vector<::ttnn::Tensor> v39 = create_inputs_for_forward_const_eval_19();
  ::std::vector<::ttnn::Tensor> v40 = forward_const_eval_19(v39);
  ::std::vector<::ttnn::Tensor> v41 = create_inputs_for_forward_const_eval_20();
  ::std::vector<::ttnn::Tensor> v42 = forward_const_eval_20(v41);
  ::std::vector<::ttnn::Tensor> v43 = create_inputs_for_forward_const_eval_21();
  ::std::vector<::ttnn::Tensor> v44 = forward_const_eval_21(v43);
  ::std::vector<::ttnn::Tensor> v45 = create_inputs_for_forward_const_eval_22();
  ::std::vector<::ttnn::Tensor> v46 = forward_const_eval_22(v45);
  ::std::vector<::ttnn::Tensor> v47 = create_inputs_for_forward_const_eval_23();
  ::std::vector<::ttnn::Tensor> v48 = forward_const_eval_23(v47);
  ::std::vector<::ttnn::Tensor> v49 = create_inputs_for_forward_const_eval_24();
  ::std::vector<::ttnn::Tensor> v50 = forward_const_eval_24(v49);
  ::std::vector<::ttnn::Tensor> v51 = create_inputs_for_forward_const_eval_25();
  ::std::vector<::ttnn::Tensor> v52 = forward_const_eval_25(v51);
  ::std::vector<::ttnn::Tensor> v53 = create_inputs_for_forward_const_eval_26();
  ::std::vector<::ttnn::Tensor> v54 = forward_const_eval_26(v53);
  ::std::vector<::ttnn::Tensor> v55 = create_inputs_for_forward_const_eval_27();
  ::std::vector<::ttnn::Tensor> v56 = forward_const_eval_27(v55);
  ::std::vector<::ttnn::Tensor> v57 = create_inputs_for_forward_const_eval_28();
  ::std::vector<::ttnn::Tensor> v58 = forward_const_eval_28(v57);
  ::std::vector<::ttnn::Tensor> v59 = create_inputs_for_forward_const_eval_29();
  ::std::vector<::ttnn::Tensor> v60 = forward_const_eval_29(v59);
  ::std::vector<::ttnn::Tensor> v61 = create_inputs_for_forward_const_eval_30();
  ::std::vector<::ttnn::Tensor> v62 = forward_const_eval_30(v61);
  ::std::vector<::ttnn::Tensor> v63 = create_inputs_for_forward_const_eval_31();
  ::std::vector<::ttnn::Tensor> v64 = forward_const_eval_31(v63);
  ::std::vector<::ttnn::Tensor> v65 = create_inputs_for_forward_const_eval_32();
  ::std::vector<::ttnn::Tensor> v66 = forward_const_eval_32(v65);
  ::std::vector<::ttnn::Tensor> v67 = create_inputs_for_forward_const_eval_33();
  ::std::vector<::ttnn::Tensor> v68 = forward_const_eval_33(v67);
  ::std::vector<::ttnn::Tensor> v69 = create_inputs_for_forward_const_eval_34();
  ::std::vector<::ttnn::Tensor> v70 = forward_const_eval_34(v69);
  ::std::vector<::ttnn::Tensor> v71 = create_inputs_for_forward_const_eval_35();
  ::std::vector<::ttnn::Tensor> v72 = forward_const_eval_35(v71);
  ::std::vector<::ttnn::Tensor> v73 = create_inputs_for_forward_const_eval_36();
  ::std::vector<::ttnn::Tensor> v74 = forward_const_eval_36(v73);
  ::std::vector<::ttnn::Tensor> v75 = create_inputs_for_forward_const_eval_37();
  ::std::vector<::ttnn::Tensor> v76 = forward_const_eval_37(v75);
  ::std::vector<::ttnn::Tensor> v77 = create_inputs_for_forward_const_eval_38();
  ::std::vector<::ttnn::Tensor> v78 = forward_const_eval_38(v77);
  ::std::vector<::ttnn::Tensor> v79 = create_inputs_for_forward_const_eval_39();
  ::std::vector<::ttnn::Tensor> v80 = forward_const_eval_39(v79);
  ::std::vector<::ttnn::Tensor> v81 = create_inputs_for_forward_const_eval_40();
  ::std::vector<::ttnn::Tensor> v82 = forward_const_eval_40(v81);
  ::std::vector<::ttnn::Tensor> v83 = create_inputs_for_forward_const_eval_41();
  ::std::vector<::ttnn::Tensor> v84 = forward_const_eval_41(v83);
  ::std::vector<::ttnn::Tensor> v85 = create_inputs_for_forward_const_eval_42();
  ::std::vector<::ttnn::Tensor> v86 = forward_const_eval_42(v85);
  ::std::vector<::ttnn::Tensor> v87 = create_inputs_for_forward_const_eval_43();
  ::std::vector<::ttnn::Tensor> v88 = forward_const_eval_43(v87);
  ::std::vector<::ttnn::Tensor> v89 = create_inputs_for_forward_const_eval_44();
  ::std::vector<::ttnn::Tensor> v90 = forward_const_eval_44(v89);
  ::std::vector<::ttnn::Tensor> v91 = create_inputs_for_forward_const_eval_45();
  ::std::vector<::ttnn::Tensor> v92 = forward_const_eval_45(v91);
  ::std::vector<::ttnn::Tensor> v93 = create_inputs_for_forward_const_eval_46();
  ::std::vector<::ttnn::Tensor> v94 = forward_const_eval_46(v93);
  ::std::vector<::ttnn::Tensor> v95 = create_inputs_for_forward_const_eval_47();
  ::std::vector<::ttnn::Tensor> v96 = forward_const_eval_47(v95);
  ::std::vector<::ttnn::Tensor> v97 = create_inputs_for_forward_const_eval_48();
  ::std::vector<::ttnn::Tensor> v98 = forward_const_eval_48(v97);
  ::std::vector<::ttnn::Tensor> v99 = create_inputs_for_forward_const_eval_49();
  ::std::vector<::ttnn::Tensor> v100 = forward_const_eval_49(v99);
  ::std::vector<::ttnn::Tensor> v101 = create_inputs_for_forward_const_eval_50();
  ::std::vector<::ttnn::Tensor> v102 = forward_const_eval_50(v101);
  ::std::vector<::ttnn::Tensor> v103 = create_inputs_for_forward_const_eval_51();
  ::std::vector<::ttnn::Tensor> v104 = forward_const_eval_51(v103);
  ::std::vector<::ttnn::Tensor> v105 = create_inputs_for_forward_const_eval_52();
  ::std::vector<::ttnn::Tensor> v106 = forward_const_eval_52(v105);
  ::std::vector<::ttnn::Tensor> v107 = create_inputs_for_forward_const_eval_53();
  ::std::vector<::ttnn::Tensor> v108 = forward_const_eval_53(v107);
  ::std::vector<::ttnn::Tensor> v109 = create_inputs_for_forward_const_eval_54();
  ::std::vector<::ttnn::Tensor> v110 = forward_const_eval_54(v109);
  ::std::vector<::ttnn::Tensor> v111 = create_inputs_for_forward_const_eval_55();
  ::std::vector<::ttnn::Tensor> v112 = forward_const_eval_55(v111);
  ::std::vector<::ttnn::Tensor> v113 = create_inputs_for_forward_const_eval_56();
  ::std::vector<::ttnn::Tensor> v114 = forward_const_eval_56(v113);
  ::std::vector<::ttnn::Tensor> v115 = create_inputs_for_forward_const_eval_57();
  ::std::vector<::ttnn::Tensor> v116 = forward_const_eval_57(v115);
  ::std::vector<::ttnn::Tensor> v117 = create_inputs_for_forward_const_eval_58();
  ::std::vector<::ttnn::Tensor> v118 = forward_const_eval_58(v117);

  // Setup the device
  //
  ttnn::MeshDevice *device = ttnn::DeviceGetter::getInstance();
  device->enable_program_cache(); // compiled kernels stored on device

  // Set print options
  //
  ttnn::set_printoptions("full");

  constexpr int NUM_OF_RUNS = 10;

  ::std::vector<::ttnn::Tensor> inputs = create_inputs_for_forward();

  time_run_wo_trace("warmup", inputs, host_input_tensor(), device);

  double total_time = 0.0;
  for (int i = 0; i < NUM_OF_RUNS; ++i) {
    total_time += time_run_wo_trace("execute-wo/trace", inputs, host_input_tensor(), device);
  }

  std::cout << "AVG: " << total_time / NUM_OF_RUNS << " seconds for run: execute-wo/trace" << std::endl;

  //====================================================================
  auto device_input = ttnn::operations::core::allocate_tensor_on_device(host_input_tensor().tensor_spec(), device);
  inputs[0] = device_input;

  const auto &[output, tid] = capture_trace(inputs, device);

  total_time = 0.0;
  for (int i = 0; i < NUM_OF_RUNS; ++i) {
    total_time += time_run_with_trace(device, tid, host_input_tensor(), device_input);
  }

  std::cout << "AVG: " << total_time / NUM_OF_RUNS << " seconds for run: execute/w-trace" << std::endl;

  int32_t v121 = 0;
  return v121;
}
