// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/device.hpp"
#include "api/tt-metalium/persistent_kernel_cache.hpp"

template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

#include "ttnn-precompiled.hpp"
::std::vector<::ttnn::Tensor> forward_const_eval_0(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 3, 64, 8, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_1(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_2(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_3(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_4(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_5(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_6(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_7(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_8(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_9(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{3136, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 512, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_10(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_11(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_12(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_13(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_14(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_15(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_16(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_17(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_18(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_19(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_20(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 1024, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_21(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_22(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_23(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_24(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_25(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 2048, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_26(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_27(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_28(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_29(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_30(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_31(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_32(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_33(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_34(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 512, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_35(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_36(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_37(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_38(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_39(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_40(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 256, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_41(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_42(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_43(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_44(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_45(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_46(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_47(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_48(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{512, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_49(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_50(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_51(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_52(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::std::nullopt, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
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
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v56 = &forward_const_eval_0;
  ::std::vector<::ttnn::Tensor> v57 = util_create_vec(v3);
  ::std::vector<::ttnn::Tensor>* v58 = &g_cached_result_forward_const_eval_0;
  ttnn::constEvalFuncWrapper(v56, v57, v58);
  ::std::vector<::ttnn::Tensor> v59 = g_cached_result_forward_const_eval_0;
  ::ttnn::Tensor v60 = v59[0];
  ttnn::deallocate(v3, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v61 = &forward_const_eval_1;
  ::std::vector<::ttnn::Tensor> v62 = util_create_vec(v26);
  ::std::vector<::ttnn::Tensor>* v63 = &g_cached_result_forward_const_eval_1;
  ttnn::constEvalFuncWrapper(v61, v62, v63);
  ::std::vector<::ttnn::Tensor> v64 = g_cached_result_forward_const_eval_1;
  ::ttnn::Tensor v65 = v64[0];
  ttnn::deallocate(v26, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v66 = &forward_const_eval_2;
  ::std::vector<::ttnn::Tensor> v67 = util_create_vec(v29);
  ::std::vector<::ttnn::Tensor>* v68 = &g_cached_result_forward_const_eval_2;
  ttnn::constEvalFuncWrapper(v66, v67, v68);
  ::std::vector<::ttnn::Tensor> v69 = g_cached_result_forward_const_eval_2;
  ::ttnn::Tensor v70 = v69[0];
  ttnn::deallocate(v29, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v71 = &forward_const_eval_3;
  ::std::vector<::ttnn::Tensor> v72 = util_create_vec(v48);
  ::std::vector<::ttnn::Tensor>* v73 = &g_cached_result_forward_const_eval_3;
  ttnn::constEvalFuncWrapper(v71, v72, v73);
  ::std::vector<::ttnn::Tensor> v74 = g_cached_result_forward_const_eval_3;
  ::ttnn::Tensor v75 = v74[0];
  ttnn::deallocate(v48, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v76 = &forward_const_eval_4;
  ::std::vector<::ttnn::Tensor> v77 = util_create_vec(v22);
  ::std::vector<::ttnn::Tensor>* v78 = &g_cached_result_forward_const_eval_4;
  ttnn::constEvalFuncWrapper(v76, v77, v78);
  ::std::vector<::ttnn::Tensor> v79 = g_cached_result_forward_const_eval_4;
  ::ttnn::Tensor v80 = v79[0];
  ttnn::deallocate(v22, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v81 = &forward_const_eval_5;
  ::std::vector<::ttnn::Tensor> v82 = util_create_vec(v34);
  ::std::vector<::ttnn::Tensor>* v83 = &g_cached_result_forward_const_eval_5;
  ttnn::constEvalFuncWrapper(v81, v82, v83);
  ::std::vector<::ttnn::Tensor> v84 = g_cached_result_forward_const_eval_5;
  ::ttnn::Tensor v85 = v84[0];
  ttnn::deallocate(v34, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v86 = &forward_const_eval_6;
  ::std::vector<::ttnn::Tensor> v87 = util_create_vec(v6);
  ::std::vector<::ttnn::Tensor>* v88 = &g_cached_result_forward_const_eval_6;
  ttnn::constEvalFuncWrapper(v86, v87, v88);
  ::std::vector<::ttnn::Tensor> v89 = g_cached_result_forward_const_eval_6;
  ::ttnn::Tensor v90 = v89[0];
  ttnn::deallocate(v6, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v91 = &forward_const_eval_7;
  ::std::vector<::ttnn::Tensor> v92 = util_create_vec(v42);
  ::std::vector<::ttnn::Tensor>* v93 = &g_cached_result_forward_const_eval_7;
  ttnn::constEvalFuncWrapper(v91, v92, v93);
  ::std::vector<::ttnn::Tensor> v94 = g_cached_result_forward_const_eval_7;
  ::ttnn::Tensor v95 = v94[0];
  ttnn::deallocate(v42, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v96 = &forward_const_eval_8;
  ::std::vector<::ttnn::Tensor> v97 = util_create_vec(v41);
  ::std::vector<::ttnn::Tensor>* v98 = &g_cached_result_forward_const_eval_8;
  ttnn::constEvalFuncWrapper(v96, v97, v98);
  ::std::vector<::ttnn::Tensor> v99 = g_cached_result_forward_const_eval_8;
  ::ttnn::Tensor v100 = v99[0];
  ttnn::deallocate(v41, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v101 = &forward_const_eval_9;
  ::std::vector<::ttnn::Tensor> v102 = util_create_vec(v17);
  ::std::vector<::ttnn::Tensor>* v103 = &g_cached_result_forward_const_eval_9;
  ttnn::constEvalFuncWrapper(v101, v102, v103);
  ::std::vector<::ttnn::Tensor> v104 = g_cached_result_forward_const_eval_9;
  ::ttnn::Tensor v105 = v104[0];
  ttnn::deallocate(v17, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v106 = &forward_const_eval_10;
  ::std::vector<::ttnn::Tensor> v107 = util_create_vec(v14);
  ::std::vector<::ttnn::Tensor>* v108 = &g_cached_result_forward_const_eval_10;
  ttnn::constEvalFuncWrapper(v106, v107, v108);
  ::std::vector<::ttnn::Tensor> v109 = g_cached_result_forward_const_eval_10;
  ::ttnn::Tensor v110 = v109[0];
  ttnn::deallocate(v14, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v111 = &forward_const_eval_11;
  ::std::vector<::ttnn::Tensor> v112 = util_create_vec(v45);
  ::std::vector<::ttnn::Tensor>* v113 = &g_cached_result_forward_const_eval_11;
  ttnn::constEvalFuncWrapper(v111, v112, v113);
  ::std::vector<::ttnn::Tensor> v114 = g_cached_result_forward_const_eval_11;
  ::ttnn::Tensor v115 = v114[0];
  ttnn::deallocate(v45, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v116 = &forward_const_eval_12;
  ::std::vector<::ttnn::Tensor> v117 = util_create_vec(v11);
  ::std::vector<::ttnn::Tensor>* v118 = &g_cached_result_forward_const_eval_12;
  ttnn::constEvalFuncWrapper(v116, v117, v118);
  ::std::vector<::ttnn::Tensor> v119 = g_cached_result_forward_const_eval_12;
  ::ttnn::Tensor v120 = v119[0];
  ttnn::deallocate(v11, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v121 = &forward_const_eval_13;
  ::std::vector<::ttnn::Tensor> v122 = util_create_vec(v18);
  ::std::vector<::ttnn::Tensor>* v123 = &g_cached_result_forward_const_eval_13;
  ttnn::constEvalFuncWrapper(v121, v122, v123);
  ::std::vector<::ttnn::Tensor> v124 = g_cached_result_forward_const_eval_13;
  ::ttnn::Tensor v125 = v124[0];
  ttnn::deallocate(v18, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v126 = &forward_const_eval_14;
  ::std::vector<::ttnn::Tensor> v127 = util_create_vec(v38);
  ::std::vector<::ttnn::Tensor>* v128 = &g_cached_result_forward_const_eval_14;
  ttnn::constEvalFuncWrapper(v126, v127, v128);
  ::std::vector<::ttnn::Tensor> v129 = g_cached_result_forward_const_eval_14;
  ::ttnn::Tensor v130 = v129[0];
  ttnn::deallocate(v38, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v131 = &forward_const_eval_15;
  ::std::vector<::ttnn::Tensor> v132 = util_create_vec(v54);
  ::std::vector<::ttnn::Tensor>* v133 = &g_cached_result_forward_const_eval_15;
  ttnn::constEvalFuncWrapper(v131, v132, v133);
  ::std::vector<::ttnn::Tensor> v134 = g_cached_result_forward_const_eval_15;
  ::ttnn::Tensor v135 = v134[0];
  ttnn::deallocate(v54, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v136 = &forward_const_eval_16;
  ::std::vector<::ttnn::Tensor> v137 = util_create_vec(v5);
  ::std::vector<::ttnn::Tensor>* v138 = &g_cached_result_forward_const_eval_16;
  ttnn::constEvalFuncWrapper(v136, v137, v138);
  ::std::vector<::ttnn::Tensor> v139 = g_cached_result_forward_const_eval_16;
  ::ttnn::Tensor v140 = v139[0];
  ttnn::deallocate(v5, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v141 = &forward_const_eval_17;
  ::std::vector<::ttnn::Tensor> v142 = util_create_vec(v25);
  ::std::vector<::ttnn::Tensor>* v143 = &g_cached_result_forward_const_eval_17;
  ttnn::constEvalFuncWrapper(v141, v142, v143);
  ::std::vector<::ttnn::Tensor> v144 = g_cached_result_forward_const_eval_17;
  ::ttnn::Tensor v145 = v144[0];
  ttnn::deallocate(v25, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v146 = &forward_const_eval_18;
  ::std::vector<::ttnn::Tensor> v147 = util_create_vec(v52);
  ::std::vector<::ttnn::Tensor>* v148 = &g_cached_result_forward_const_eval_18;
  ttnn::constEvalFuncWrapper(v146, v147, v148);
  ::std::vector<::ttnn::Tensor> v149 = g_cached_result_forward_const_eval_18;
  ::ttnn::Tensor v150 = v149[0];
  ttnn::deallocate(v52, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v151 = &forward_const_eval_19;
  ::std::vector<::ttnn::Tensor> v152 = util_create_vec(v12);
  ::std::vector<::ttnn::Tensor>* v153 = &g_cached_result_forward_const_eval_19;
  ttnn::constEvalFuncWrapper(v151, v152, v153);
  ::std::vector<::ttnn::Tensor> v154 = g_cached_result_forward_const_eval_19;
  ::ttnn::Tensor v155 = v154[0];
  ttnn::deallocate(v12, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v156 = &forward_const_eval_20;
  ::std::vector<::ttnn::Tensor> v157 = util_create_vec(v30);
  ::std::vector<::ttnn::Tensor>* v158 = &g_cached_result_forward_const_eval_20;
  ttnn::constEvalFuncWrapper(v156, v157, v158);
  ::std::vector<::ttnn::Tensor> v159 = g_cached_result_forward_const_eval_20;
  ::ttnn::Tensor v160 = v159[0];
  ttnn::deallocate(v30, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v161 = &forward_const_eval_21;
  ::std::vector<::ttnn::Tensor> v162 = util_create_vec(v19);
  ::std::vector<::ttnn::Tensor>* v163 = &g_cached_result_forward_const_eval_21;
  ttnn::constEvalFuncWrapper(v161, v162, v163);
  ::std::vector<::ttnn::Tensor> v164 = g_cached_result_forward_const_eval_21;
  ::ttnn::Tensor v165 = v164[0];
  ttnn::deallocate(v19, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v166 = &forward_const_eval_22;
  ::std::vector<::ttnn::Tensor> v167 = util_create_vec(v20);
  ::std::vector<::ttnn::Tensor>* v168 = &g_cached_result_forward_const_eval_22;
  ttnn::constEvalFuncWrapper(v166, v167, v168);
  ::std::vector<::ttnn::Tensor> v169 = g_cached_result_forward_const_eval_22;
  ::ttnn::Tensor v170 = v169[0];
  ttnn::deallocate(v20, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v171 = &forward_const_eval_23;
  ::std::vector<::ttnn::Tensor> v172 = util_create_vec(v39);
  ::std::vector<::ttnn::Tensor>* v173 = &g_cached_result_forward_const_eval_23;
  ttnn::constEvalFuncWrapper(v171, v172, v173);
  ::std::vector<::ttnn::Tensor> v174 = g_cached_result_forward_const_eval_23;
  ::ttnn::Tensor v175 = v174[0];
  ttnn::deallocate(v39, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v176 = &forward_const_eval_24;
  ::std::vector<::ttnn::Tensor> v177 = util_create_vec(v40);
  ::std::vector<::ttnn::Tensor>* v178 = &g_cached_result_forward_const_eval_24;
  ttnn::constEvalFuncWrapper(v176, v177, v178);
  ::std::vector<::ttnn::Tensor> v179 = g_cached_result_forward_const_eval_24;
  ::ttnn::Tensor v180 = v179[0];
  ttnn::deallocate(v40, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v181 = &forward_const_eval_25;
  ::std::vector<::ttnn::Tensor> v182 = util_create_vec(v49);
  ::std::vector<::ttnn::Tensor>* v183 = &g_cached_result_forward_const_eval_25;
  ttnn::constEvalFuncWrapper(v181, v182, v183);
  ::std::vector<::ttnn::Tensor> v184 = g_cached_result_forward_const_eval_25;
  ::ttnn::Tensor v185 = v184[0];
  ttnn::deallocate(v49, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v186 = &forward_const_eval_26;
  ::std::vector<::ttnn::Tensor> v187 = util_create_vec(v13);
  ::std::vector<::ttnn::Tensor>* v188 = &g_cached_result_forward_const_eval_26;
  ttnn::constEvalFuncWrapper(v186, v187, v188);
  ::std::vector<::ttnn::Tensor> v189 = g_cached_result_forward_const_eval_26;
  ::ttnn::Tensor v190 = v189[0];
  ttnn::deallocate(v13, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v191 = &forward_const_eval_27;
  ::std::vector<::ttnn::Tensor> v192 = util_create_vec(v33);
  ::std::vector<::ttnn::Tensor>* v193 = &g_cached_result_forward_const_eval_27;
  ttnn::constEvalFuncWrapper(v191, v192, v193);
  ::std::vector<::ttnn::Tensor> v194 = g_cached_result_forward_const_eval_27;
  ::ttnn::Tensor v195 = v194[0];
  ttnn::deallocate(v33, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v196 = &forward_const_eval_28;
  ::std::vector<::ttnn::Tensor> v197 = util_create_vec(v53);
  ::std::vector<::ttnn::Tensor>* v198 = &g_cached_result_forward_const_eval_28;
  ttnn::constEvalFuncWrapper(v196, v197, v198);
  ::std::vector<::ttnn::Tensor> v199 = g_cached_result_forward_const_eval_28;
  ::ttnn::Tensor v200 = v199[0];
  ttnn::deallocate(v53, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v201 = &forward_const_eval_29;
  ::std::vector<::ttnn::Tensor> v202 = util_create_vec(v4);
  ::std::vector<::ttnn::Tensor>* v203 = &g_cached_result_forward_const_eval_29;
  ttnn::constEvalFuncWrapper(v201, v202, v203);
  ::std::vector<::ttnn::Tensor> v204 = g_cached_result_forward_const_eval_29;
  ::ttnn::Tensor v205 = v204[0];
  ttnn::deallocate(v4, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v206 = &forward_const_eval_30;
  ::std::vector<::ttnn::Tensor> v207 = util_create_vec(v55);
  ::std::vector<::ttnn::Tensor>* v208 = &g_cached_result_forward_const_eval_30;
  ttnn::constEvalFuncWrapper(v206, v207, v208);
  ::std::vector<::ttnn::Tensor> v209 = g_cached_result_forward_const_eval_30;
  ::ttnn::Tensor v210 = v209[0];
  ttnn::deallocate(v55, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v211 = &forward_const_eval_31;
  ::std::vector<::ttnn::Tensor> v212 = util_create_vec(v9);
  ::std::vector<::ttnn::Tensor>* v213 = &g_cached_result_forward_const_eval_31;
  ttnn::constEvalFuncWrapper(v211, v212, v213);
  ::std::vector<::ttnn::Tensor> v214 = g_cached_result_forward_const_eval_31;
  ::ttnn::Tensor v215 = v214[0];
  ttnn::deallocate(v9, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v216 = &forward_const_eval_32;
  ::std::vector<::ttnn::Tensor> v217 = util_create_vec(v24);
  ::std::vector<::ttnn::Tensor>* v218 = &g_cached_result_forward_const_eval_32;
  ttnn::constEvalFuncWrapper(v216, v217, v218);
  ::std::vector<::ttnn::Tensor> v219 = g_cached_result_forward_const_eval_32;
  ::ttnn::Tensor v220 = v219[0];
  ttnn::deallocate(v24, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v221 = &forward_const_eval_33;
  ::std::vector<::ttnn::Tensor> v222 = util_create_vec(v32);
  ::std::vector<::ttnn::Tensor>* v223 = &g_cached_result_forward_const_eval_33;
  ttnn::constEvalFuncWrapper(v221, v222, v223);
  ::std::vector<::ttnn::Tensor> v224 = g_cached_result_forward_const_eval_33;
  ::ttnn::Tensor v225 = v224[0];
  ttnn::deallocate(v32, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v226 = &forward_const_eval_34;
  ::std::vector<::ttnn::Tensor> v227 = util_create_vec(v46);
  ::std::vector<::ttnn::Tensor>* v228 = &g_cached_result_forward_const_eval_34;
  ttnn::constEvalFuncWrapper(v226, v227, v228);
  ::std::vector<::ttnn::Tensor> v229 = g_cached_result_forward_const_eval_34;
  ::ttnn::Tensor v230 = v229[0];
  ttnn::deallocate(v46, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v231 = &forward_const_eval_35;
  ::std::vector<::ttnn::Tensor> v232 = util_create_vec(v44);
  ::std::vector<::ttnn::Tensor>* v233 = &g_cached_result_forward_const_eval_35;
  ttnn::constEvalFuncWrapper(v231, v232, v233);
  ::std::vector<::ttnn::Tensor> v234 = g_cached_result_forward_const_eval_35;
  ::ttnn::Tensor v235 = v234[0];
  ttnn::deallocate(v44, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v236 = &forward_const_eval_36;
  ::std::vector<::ttnn::Tensor> v237 = util_create_vec(v47);
  ::std::vector<::ttnn::Tensor>* v238 = &g_cached_result_forward_const_eval_36;
  ttnn::constEvalFuncWrapper(v236, v237, v238);
  ::std::vector<::ttnn::Tensor> v239 = g_cached_result_forward_const_eval_36;
  ::ttnn::Tensor v240 = v239[0];
  ttnn::deallocate(v47, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v241 = &forward_const_eval_37;
  ::std::vector<::ttnn::Tensor> v242 = util_create_vec(v21);
  ::std::vector<::ttnn::Tensor>* v243 = &g_cached_result_forward_const_eval_37;
  ttnn::constEvalFuncWrapper(v241, v242, v243);
  ::std::vector<::ttnn::Tensor> v244 = g_cached_result_forward_const_eval_37;
  ::ttnn::Tensor v245 = v244[0];
  ttnn::deallocate(v21, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v246 = &forward_const_eval_38;
  ::std::vector<::ttnn::Tensor> v247 = util_create_vec(v51);
  ::std::vector<::ttnn::Tensor>* v248 = &g_cached_result_forward_const_eval_38;
  ttnn::constEvalFuncWrapper(v246, v247, v248);
  ::std::vector<::ttnn::Tensor> v249 = g_cached_result_forward_const_eval_38;
  ::ttnn::Tensor v250 = v249[0];
  ttnn::deallocate(v51, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v251 = &forward_const_eval_39;
  ::std::vector<::ttnn::Tensor> v252 = util_create_vec(v8);
  ::std::vector<::ttnn::Tensor>* v253 = &g_cached_result_forward_const_eval_39;
  ttnn::constEvalFuncWrapper(v251, v252, v253);
  ::std::vector<::ttnn::Tensor> v254 = g_cached_result_forward_const_eval_39;
  ::ttnn::Tensor v255 = v254[0];
  ttnn::deallocate(v8, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v256 = &forward_const_eval_40;
  ::std::vector<::ttnn::Tensor> v257 = util_create_vec(v27);
  ::std::vector<::ttnn::Tensor>* v258 = &g_cached_result_forward_const_eval_40;
  ttnn::constEvalFuncWrapper(v256, v257, v258);
  ::std::vector<::ttnn::Tensor> v259 = g_cached_result_forward_const_eval_40;
  ::ttnn::Tensor v260 = v259[0];
  ttnn::deallocate(v27, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v261 = &forward_const_eval_41;
  ::std::vector<::ttnn::Tensor> v262 = util_create_vec(v31);
  ::std::vector<::ttnn::Tensor>* v263 = &g_cached_result_forward_const_eval_41;
  ttnn::constEvalFuncWrapper(v261, v262, v263);
  ::std::vector<::ttnn::Tensor> v264 = g_cached_result_forward_const_eval_41;
  ::ttnn::Tensor v265 = v264[0];
  ttnn::deallocate(v31, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v266 = &forward_const_eval_42;
  ::std::vector<::ttnn::Tensor> v267 = util_create_vec(v7);
  ::std::vector<::ttnn::Tensor>* v268 = &g_cached_result_forward_const_eval_42;
  ttnn::constEvalFuncWrapper(v266, v267, v268);
  ::std::vector<::ttnn::Tensor> v269 = g_cached_result_forward_const_eval_42;
  ::ttnn::Tensor v270 = v269[0];
  ttnn::deallocate(v7, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v271 = &forward_const_eval_43;
  ::std::vector<::ttnn::Tensor> v272 = util_create_vec(v28);
  ::std::vector<::ttnn::Tensor>* v273 = &g_cached_result_forward_const_eval_43;
  ttnn::constEvalFuncWrapper(v271, v272, v273);
  ::std::vector<::ttnn::Tensor> v274 = g_cached_result_forward_const_eval_43;
  ::ttnn::Tensor v275 = v274[0];
  ttnn::deallocate(v28, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v276 = &forward_const_eval_44;
  ::std::vector<::ttnn::Tensor> v277 = util_create_vec(v35);
  ::std::vector<::ttnn::Tensor>* v278 = &g_cached_result_forward_const_eval_44;
  ttnn::constEvalFuncWrapper(v276, v277, v278);
  ::std::vector<::ttnn::Tensor> v279 = g_cached_result_forward_const_eval_44;
  ::ttnn::Tensor v280 = v279[0];
  ttnn::deallocate(v35, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v281 = &forward_const_eval_45;
  ::std::vector<::ttnn::Tensor> v282 = util_create_vec(v50);
  ::std::vector<::ttnn::Tensor>* v283 = &g_cached_result_forward_const_eval_45;
  ttnn::constEvalFuncWrapper(v281, v282, v283);
  ::std::vector<::ttnn::Tensor> v284 = g_cached_result_forward_const_eval_45;
  ::ttnn::Tensor v285 = v284[0];
  ttnn::deallocate(v50, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v286 = &forward_const_eval_46;
  ::std::vector<::ttnn::Tensor> v287 = util_create_vec(v37);
  ::std::vector<::ttnn::Tensor>* v288 = &g_cached_result_forward_const_eval_46;
  ttnn::constEvalFuncWrapper(v286, v287, v288);
  ::std::vector<::ttnn::Tensor> v289 = g_cached_result_forward_const_eval_46;
  ::ttnn::Tensor v290 = v289[0];
  ttnn::deallocate(v37, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v291 = &forward_const_eval_47;
  ::std::vector<::ttnn::Tensor> v292 = util_create_vec(v43);
  ::std::vector<::ttnn::Tensor>* v293 = &g_cached_result_forward_const_eval_47;
  ttnn::constEvalFuncWrapper(v291, v292, v293);
  ::std::vector<::ttnn::Tensor> v294 = g_cached_result_forward_const_eval_47;
  ::ttnn::Tensor v295 = v294[0];
  ttnn::deallocate(v43, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v296 = &forward_const_eval_48;
  ::std::vector<::ttnn::Tensor> v297 = util_create_vec(v15);
  ::std::vector<::ttnn::Tensor>* v298 = &g_cached_result_forward_const_eval_48;
  ttnn::constEvalFuncWrapper(v296, v297, v298);
  ::std::vector<::ttnn::Tensor> v299 = g_cached_result_forward_const_eval_48;
  ::ttnn::Tensor v300 = v299[0];
  ttnn::deallocate(v15, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v301 = &forward_const_eval_49;
  ::std::vector<::ttnn::Tensor> v302 = util_create_vec(v16);
  ::std::vector<::ttnn::Tensor>* v303 = &g_cached_result_forward_const_eval_49;
  ttnn::constEvalFuncWrapper(v301, v302, v303);
  ::std::vector<::ttnn::Tensor> v304 = g_cached_result_forward_const_eval_49;
  ::ttnn::Tensor v305 = v304[0];
  ttnn::deallocate(v16, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v306 = &forward_const_eval_50;
  ::std::vector<::ttnn::Tensor> v307 = util_create_vec(v10);
  ::std::vector<::ttnn::Tensor>* v308 = &g_cached_result_forward_const_eval_50;
  ttnn::constEvalFuncWrapper(v306, v307, v308);
  ::std::vector<::ttnn::Tensor> v309 = g_cached_result_forward_const_eval_50;
  ::ttnn::Tensor v310 = v309[0];
  ttnn::deallocate(v10, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v311 = &forward_const_eval_51;
  ::std::vector<::ttnn::Tensor> v312 = util_create_vec(v23);
  ::std::vector<::ttnn::Tensor>* v313 = &g_cached_result_forward_const_eval_51;
  ttnn::constEvalFuncWrapper(v311, v312, v313);
  ::std::vector<::ttnn::Tensor> v314 = g_cached_result_forward_const_eval_51;
  ::ttnn::Tensor v315 = v314[0];
  ttnn::deallocate(v23, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v316 = &forward_const_eval_52;
  ::std::vector<::ttnn::Tensor> v317 = util_create_vec(v36);
  ::std::vector<::ttnn::Tensor>* v318 = &g_cached_result_forward_const_eval_52;
  ttnn::constEvalFuncWrapper(v316, v317, v318);
  ::std::vector<::ttnn::Tensor> v319 = g_cached_result_forward_const_eval_52;
  ::ttnn::Tensor v320 = v319[0];
  ttnn::deallocate(v36, false);
  ttnn::distributed::MeshDevice* v321 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v322 = ::std::get<0>(ttnn::conv2d(v2, v60, v321, 3, 64, 8, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v60, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v323 = ttnn::to_layout(v322, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v322, false);
  ::ttnn::Tensor v324 = ttnn::max_pool2d(v323, 8, 112, 112, 64, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::std::nullopt, false);
  ttnn::deallocate(v323, false);
  ::ttnn::Tensor v325 = ttnn::to_layout(v324, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v324, false);
  ::ttnn::Tensor v326 = ttnn::to_memory_config(v325, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v327 = ::std::get<0>(ttnn::conv2d(v326, v270, v321, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v326, false);
  ttnn::deallocate(v270, false);
  ::ttnn::Tensor v328 = ttnn::to_memory_config(v327, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v327, false);
  ::ttnn::Tensor v329 = ttnn::to_memory_config(v325, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v325, false);
  ::ttnn::Tensor v330 = ::std::get<0>(ttnn::conv2d(v329, v205, v321, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v329, false);
  ttnn::deallocate(v205, false);
  ::ttnn::Tensor v331 = ::std::get<0>(ttnn::conv2d(v330, v140, v321, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v330, false);
  ttnn::deallocate(v140, false);
  ::ttnn::Tensor v332 = ::std::get<0>(ttnn::conv2d(v331, v90, v321, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v331, false);
  ttnn::deallocate(v90, false);
  ::ttnn::Tensor v333 = ttnn::add(v332, v328, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v332, false);
  ttnn::deallocate(v328, false);
  ::ttnn::Tensor v334 = ttnn::relu(v333, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v333, false);
  ::ttnn::Tensor v335 = ttnn::to_memory_config(v334, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v334, false);
  ::ttnn::Tensor v336 = ::std::get<0>(ttnn::conv2d(v335, v255, v321, 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v255, false);
  ::ttnn::Tensor v337 = ::std::get<0>(ttnn::conv2d(v336, v215, v321, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v336, false);
  ttnn::deallocate(v215, false);
  ::ttnn::Tensor v338 = ::std::get<0>(ttnn::conv2d(v337, v310, v321, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v337, false);
  ttnn::deallocate(v310, false);
  ::ttnn::Tensor v339 = ttnn::add(v335, v338, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v338, false);
  ttnn::deallocate(v335, false);
  ::ttnn::Tensor v340 = ttnn::relu(v339, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v339, false);
  ::ttnn::Tensor v341 = ::std::get<0>(ttnn::conv2d(v340, v120, v321, 256, 64, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v120, false);
  ::ttnn::Tensor v342 = ::std::get<0>(ttnn::conv2d(v341, v155, v321, 64, 64, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v341, false);
  ttnn::deallocate(v155, false);
  ::ttnn::Tensor v343 = ::std::get<0>(ttnn::conv2d(v342, v190, v321, 64, 256, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v342, false);
  ttnn::deallocate(v190, false);
  ::ttnn::Tensor v344 = ttnn::add(v340, v343, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v343, false);
  ttnn::deallocate(v340, false);
  ::ttnn::Tensor v345 = ttnn::relu(v344, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v344, false);
  ::ttnn::Tensor v346 = ttnn::to_memory_config(v345, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{3136, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v347 = ::std::get<0>(ttnn::conv2d(v346, v105, v321, 256, 512, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v346, false);
  ttnn::deallocate(v105, false);
  ::ttnn::Tensor v348 = ttnn::to_memory_config(v347, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v347, false);
  ::ttnn::Tensor v349 = ttnn::to_memory_config(v345, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v345, false);
  ::ttnn::Tensor v350 = ::std::get<0>(ttnn::conv2d(v349, v110, v321, 256, 128, 8, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{4, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v349, false);
  ttnn::deallocate(v110, false);
  ::ttnn::Tensor v351 = ttnn::to_memory_config(v350, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{512, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v350, false);
  ::ttnn::Tensor v352 = ::std::get<0>(ttnn::conv2d(v351, v300, v321, 128, 128, 8, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v351, false);
  ttnn::deallocate(v300, false);
  ::ttnn::Tensor v353 = ::std::get<0>(ttnn::conv2d(v352, v305, v321, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v352, false);
  ttnn::deallocate(v305, false);
  ::ttnn::Tensor v354 = ttnn::add(v353, v348, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v353, false);
  ttnn::deallocate(v348, false);
  ::ttnn::Tensor v355 = ttnn::relu(v354, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 5}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 6}, ::ttnn::CoreCoord{0, 6}}}}, ::std::array<uint32_t, 2>{128, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v354, false);
  ::ttnn::Tensor v356 = ttnn::to_memory_config(v355, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v355, false);
  ::ttnn::Tensor v357 = ::std::get<0>(ttnn::conv2d(v356, v125, v321, 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v125, false);
  ::ttnn::Tensor v358 = ::std::get<0>(ttnn::conv2d(v357, v165, v321, 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v357, false);
  ttnn::deallocate(v165, false);
  ::ttnn::Tensor v359 = ::std::get<0>(ttnn::conv2d(v358, v170, v321, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v358, false);
  ttnn::deallocate(v170, false);
  ::ttnn::Tensor v360 = ttnn::add(v356, v359, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v359, false);
  ttnn::deallocate(v356, false);
  ::ttnn::Tensor v361 = ttnn::relu(v360, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v360, false);
  ::ttnn::Tensor v362 = ::std::get<0>(ttnn::conv2d(v361, v245, v321, 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v245, false);
  ::ttnn::Tensor v363 = ::std::get<0>(ttnn::conv2d(v362, v80, v321, 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v362, false);
  ttnn::deallocate(v80, false);
  ::ttnn::Tensor v364 = ::std::get<0>(ttnn::conv2d(v363, v315, v321, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v363, false);
  ttnn::deallocate(v315, false);
  ::ttnn::Tensor v365 = ttnn::add(v361, v364, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v364, false);
  ttnn::deallocate(v361, false);
  ::ttnn::Tensor v366 = ttnn::relu(v365, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v365, false);
  ::ttnn::Tensor v367 = ::std::get<0>(ttnn::conv2d(v366, v220, v321, 512, 128, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v220, false);
  ::ttnn::Tensor v368 = ::std::get<0>(ttnn::conv2d(v367, v145, v321, 128, 128, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v367, false);
  ttnn::deallocate(v145, false);
  ::ttnn::Tensor v369 = ::std::get<0>(ttnn::conv2d(v368, v65, v321, 128, 512, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v368, false);
  ttnn::deallocate(v65, false);
  ::ttnn::Tensor v370 = ttnn::add(v366, v369, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v369, false);
  ttnn::deallocate(v366, false);
  ::ttnn::Tensor v371 = ttnn::relu(v370, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v370, false);
  ::ttnn::Tensor v372 = ttnn::to_memory_config(v371, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v373 = ::std::get<0>(ttnn::conv2d(v372, v160, v321, 512, 1024, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v372, false);
  ttnn::deallocate(v160, false);
  ::ttnn::Tensor v374 = ttnn::to_memory_config(v373, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v373, false);
  ::ttnn::Tensor v375 = ttnn::to_memory_config(v371, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v371, false);
  ::ttnn::Tensor v376 = ::std::get<0>(ttnn::conv2d(v375, v260, v321, 512, 256, 8, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{800, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v375, false);
  ttnn::deallocate(v260, false);
  ::ttnn::Tensor v377 = ttnn::to_memory_config(v376, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v376, false);
  ::ttnn::Tensor v378 = ::std::get<0>(ttnn::conv2d(v377, v275, v321, 256, 256, 8, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v377, false);
  ttnn::deallocate(v275, false);
  ::ttnn::Tensor v379 = ::std::get<0>(ttnn::conv2d(v378, v70, v321, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v378, false);
  ttnn::deallocate(v70, false);
  ::ttnn::Tensor v380 = ttnn::add(v379, v374, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v379, false);
  ttnn::deallocate(v374, false);
  ::ttnn::Tensor v381 = ttnn::relu(v380, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v380, false);
  ::ttnn::Tensor v382 = ttnn::to_memory_config(v381, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v381, false);
  ::ttnn::Tensor v383 = ::std::get<0>(ttnn::conv2d(v382, v265, v321, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v265, false);
  ::ttnn::Tensor v384 = ::std::get<0>(ttnn::conv2d(v383, v225, v321, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v383, false);
  ttnn::deallocate(v225, false);
  ::ttnn::Tensor v385 = ::std::get<0>(ttnn::conv2d(v384, v195, v321, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v384, false);
  ttnn::deallocate(v195, false);
  ::ttnn::Tensor v386 = ttnn::add(v382, v385, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v385, false);
  ttnn::deallocate(v382, false);
  ::ttnn::Tensor v387 = ttnn::relu(v386, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v386, false);
  ::ttnn::Tensor v388 = ::std::get<0>(ttnn::conv2d(v387, v85, v321, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v85, false);
  ::ttnn::Tensor v389 = ::std::get<0>(ttnn::conv2d(v388, v280, v321, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v388, false);
  ttnn::deallocate(v280, false);
  ::ttnn::Tensor v390 = ::std::get<0>(ttnn::conv2d(v389, v320, v321, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v389, false);
  ttnn::deallocate(v320, false);
  ::ttnn::Tensor v391 = ttnn::add(v387, v390, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v390, false);
  ttnn::deallocate(v387, false);
  ::ttnn::Tensor v392 = ttnn::relu(v391, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v391, false);
  ::ttnn::Tensor v393 = ::std::get<0>(ttnn::conv2d(v392, v290, v321, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v290, false);
  ::ttnn::Tensor v394 = ::std::get<0>(ttnn::conv2d(v393, v130, v321, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v393, false);
  ttnn::deallocate(v130, false);
  ::ttnn::Tensor v395 = ::std::get<0>(ttnn::conv2d(v394, v175, v321, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v394, false);
  ttnn::deallocate(v175, false);
  ::ttnn::Tensor v396 = ttnn::add(v392, v395, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v395, false);
  ttnn::deallocate(v392, false);
  ::ttnn::Tensor v397 = ttnn::relu(v396, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v396, false);
  ::ttnn::Tensor v398 = ::std::get<0>(ttnn::conv2d(v397, v180, v321, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v180, false);
  ::ttnn::Tensor v399 = ::std::get<0>(ttnn::conv2d(v398, v100, v321, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v398, false);
  ttnn::deallocate(v100, false);
  ::ttnn::Tensor v400 = ::std::get<0>(ttnn::conv2d(v399, v95, v321, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v399, false);
  ttnn::deallocate(v95, false);
  ::ttnn::Tensor v401 = ttnn::add(v397, v400, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v400, false);
  ttnn::deallocate(v397, false);
  ::ttnn::Tensor v402 = ttnn::relu(v401, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v401, false);
  ::ttnn::Tensor v403 = ::std::get<0>(ttnn::conv2d(v402, v295, v321, 1024, 256, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v295, false);
  ::ttnn::Tensor v404 = ::std::get<0>(ttnn::conv2d(v403, v235, v321, 256, 256, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v403, false);
  ttnn::deallocate(v235, false);
  ::ttnn::Tensor v405 = ::std::get<0>(ttnn::conv2d(v404, v115, v321, 256, 1024, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v404, false);
  ttnn::deallocate(v115, false);
  ::ttnn::Tensor v406 = ttnn::add(v402, v405, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v405, false);
  ttnn::deallocate(v402, false);
  ::ttnn::Tensor v407 = ttnn::relu(v406, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v406, false);
  ::ttnn::Tensor v408 = ttnn::to_memory_config(v407, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v409 = ::std::get<0>(ttnn::conv2d(v408, v185, v321, 1024, 2048, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v408, false);
  ttnn::deallocate(v185, false);
  ::ttnn::Tensor v410 = ttnn::to_memory_config(v409, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v409, false);
  ::ttnn::Tensor v411 = ttnn::to_memory_config(v407, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v407, false);
  ::ttnn::Tensor v412 = ::std::get<0>(ttnn::conv2d(v411, v230, v321, 1024, 512, 8, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v411, false);
  ttnn::deallocate(v230, false);
  ::ttnn::Tensor v413 = ::std::get<0>(ttnn::conv2d(v412, v240, v321, 512, 512, 8, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v412, false);
  ttnn::deallocate(v240, false);
  ::ttnn::Tensor v414 = ::std::get<0>(ttnn::conv2d(v413, v75, v321, 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v413, false);
  ttnn::deallocate(v75, false);
  ::ttnn::Tensor v415 = ttnn::add(v414, v410, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v414, false);
  ttnn::deallocate(v410, false);
  ::ttnn::Tensor v416 = ttnn::relu(v415, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{64, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v415, false);
  ::ttnn::Tensor v417 = ttnn::to_memory_config(v416, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v416, false);
  ::ttnn::Tensor v418 = ::std::get<0>(ttnn::conv2d(v417, v285, v321, 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v285, false);
  ::ttnn::Tensor v419 = ::std::get<0>(ttnn::conv2d(v418, v250, v321, 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v418, false);
  ttnn::deallocate(v250, false);
  ::ttnn::Tensor v420 = ::std::get<0>(ttnn::conv2d(v419, v150, v321, 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v419, false);
  ttnn::deallocate(v150, false);
  ::ttnn::Tensor v421 = ttnn::add(v417, v420, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v420, false);
  ttnn::deallocate(v417, false);
  ::ttnn::Tensor v422 = ttnn::relu(v421, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v421, false);
  ::ttnn::Tensor v423 = ::std::get<0>(ttnn::conv2d(v422, v200, v321, 2048, 512, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v200, false);
  ::ttnn::Tensor v424 = ::std::get<0>(ttnn::conv2d(v423, v135, v321, 512, 512, 8, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16, .activation = "relu", .deallocate_activation = false, .reallocate_halo_output = true, .act_block_h_override = 0, .act_block_w_div = 1, .reshard_if_not_optimal = false, .override_sharding_config = false, .transpose_shards = true, .output_layout = ::ttnn::Layout::TILE, .preprocess_weights_on_device = false}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v423, false);
  ttnn::deallocate(v135, false);
  ::ttnn::Tensor v425 = ::std::get<0>(ttnn::conv2d(v424, v210, v321, 512, 2048, 8, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v424, false);
  ttnn::deallocate(v210, false);
  ::ttnn::Tensor v426 = ttnn::add(v422, v425, ::ttnn::DataType::BFLOAT16, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v425, false);
  ttnn::deallocate(v422, false);
  ::ttnn::Tensor v427 = ttnn::relu(v426, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v426, false);
  ::ttnn::Tensor v428 = ttnn::avg_pool2d(v427, 8, 7, 7, 2048, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{0, 0}, false, true, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::std::nullopt, false);
  ttnn::deallocate(v427, false);
  ::std::vector<::ttnn::Tensor> v429 = util_create_vec(v428);
  return v429;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_0() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 3, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_1() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_2() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_3() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_4() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_5() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_6() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_7() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_8() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_9() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_10() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_11() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_12() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_13() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_14() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_15() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_16() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_17() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_18() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_19() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_20() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_21() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_22() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_23() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_24() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_25() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_26() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_27() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_28() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_29() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_30() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_31() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_32() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_33() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_34() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_35() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_36() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_37() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_38() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_39() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_40() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_41() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_42() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_43() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_44() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_45() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_46() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_47() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_48() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_49() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_50() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_51() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward_const_eval_52() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v2 = util_create_vec(v1);
  return v2;
}


double time_run(const std::string &run_name, std::vector<::ttnn::Tensor> &inputs, ttnn::MeshDevice *device) {
  auto start = std::chrono::high_resolution_clock::now();
  ::std::vector<::ttnn::Tensor> v2 = forward(inputs);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // Print results
  //
  std::cout << duration.count() << " seconds for run: " << run_name << std::endl;
  auto batch_size = 8;
  std::cout << "  FPS: " << batch_size / duration.count() << std::endl;

  return duration.count();
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({1, 1, 401408, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({64, 3, 7, 7}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::ones(::ttnn::Shape({64, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::ones(::ttnn::Shape({64, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::ones(::ttnn::Shape({64, 64, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::ones(::ttnn::Shape({256, 64, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::ones(::ttnn::Shape({128, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::ones(::ttnn::Shape({512, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::ones(::ttnn::Shape({128, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::ones(::ttnn::Shape({128, 128, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::ones(::ttnn::Shape({512, 128, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::ones(::ttnn::Shape({256, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::ones(::ttnn::Shape({1024, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::ones(::ttnn::Shape({256, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::ones(::ttnn::Shape({256, 256, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::ones(::ttnn::Shape({1024, 256, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::ones(::ttnn::Shape({512, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::ones(::ttnn::Shape({2048, 1024, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::ones(::ttnn::Shape({512, 2048, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::ones(::ttnn::Shape({512, 512, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::ones(::ttnn::Shape({2048, 512, 1, 1}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v57 = util_create_vec(v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56);
  return v57;
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
  ::std::vector<::ttnn::Tensor> v107 = create_inputs_for_forward();

  ttnn::MeshDevice *device = ttnn::DeviceGetter::getInstance();

  tt::tt_metal::detail::EnablePersistentKernelCache();

  time_run("first", v107, device);

  device->enable_program_cache();
  time_run("second", v107, device);
  time_run("third", v107, device);
  time_run("fourth", v107, device);
  int32_t v109 = 0;
  return v109;
}
