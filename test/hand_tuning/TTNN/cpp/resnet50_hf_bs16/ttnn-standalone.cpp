#include "ttnn-precompiled.hpp"
::std::vector<::ttnn::Tensor> forward_const_eval_0(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 1024, 512, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_1(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_2(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_3(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v2, ::std::vector<int32_t>{1, 1000}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_4(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_5(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_6(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_7(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_8(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_9(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_10(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_11(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 16, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_12(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ::ttnn::Tensor v6 = ttnn::reshape(v5, ::std::vector<int32_t>{2048, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::to_layout(v7, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::multiply(v8, v6, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::from_device(v10);
  ttnn::deallocate(v10, false);
  ::ttnn::Tensor v12 = ttnn::operations::conv::conv2d::prepare_conv_weights(v11, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v11, false);
  ::std::vector<::ttnn::Tensor> v13 = util_create_vec(v12);
  return v13;
}

::std::vector<::ttnn::Tensor> forward_const_eval_13(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_14(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_15(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_16(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{448, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_17(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_18(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_19(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_20(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, 3, 64, 16, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{3, 3, 3, 3}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_21(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_22(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_23(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_24(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_25(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_26(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_27(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_28(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, 256, 512, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_29(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_30(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v2, ::std::vector<int32_t>{1, 1, 2048, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::permute(v4, ::ttsl::SmallVector<int64_t>{0, 1, 3, 2}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::repeat(v5, ::ttnn::Shape({16, 1, 49, 1}));
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::reshape(v6, ::std::vector<int32_t>{1, 1, 784, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}

::std::vector<::ttnn::Tensor> forward_const_eval_31(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_32(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_33(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 128, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_34(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_35(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_36(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_37(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_38(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ::ttnn::Tensor v6 = ttnn::reshape(v5, ::std::vector<int32_t>{2048, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::to_layout(v7, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v9 = ttnn::multiply(v8, v6, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v10 = ttnn::to_layout(v9, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::from_device(v10);
  ttnn::deallocate(v10, false);
  ::ttnn::Tensor v12 = ttnn::operations::conv::conv2d::prepare_conv_weights(v11, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v11, false);
  ::std::vector<::ttnn::Tensor> v13 = util_create_vec(v12);
  return v13;
}

::std::vector<::ttnn::Tensor> forward_const_eval_39(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_40(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 2048, 512, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_41(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_42(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_43(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_44(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_45(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::operations::conv::conv2d::prepare_conv_weights(v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 2048, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_46(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_47(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 512, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_48(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 3, 64, 16, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{3, 3, 3, 3}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_49(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_50(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 256, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_51(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_52(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_53(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_54(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_55(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_56(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_57(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_58(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_59(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 128, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_60(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, 2048, 512, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_61(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_62(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_63(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_64(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_65(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_66(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 1024, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_67(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_68(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_69(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, 2048, 512, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_70(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 512, 16, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_71(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{448, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 512, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_72(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_73(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 2048, 512, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_74(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 1024, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_75(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_76(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_77(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_78(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ::std::vector<::ttnn::Tensor> v5 = util_create_vec(v4);
  return v5;
}

::std::vector<::ttnn::Tensor> forward_const_eval_79(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_80(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ::ttnn::Tensor v6 = ttnn::reshape(v5, ::std::vector<int32_t>{2048, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::permute(v6, ::ttsl::SmallVector<int64_t>{2, 3, 0, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::permute(v9, ::ttsl::SmallVector<int64_t>{2, 3, 0, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::multiply(v10, v7, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v10, false);
  ttnn::deallocate(v7, false);
  ::ttnn::Tensor v12 = ttnn::permute(v11, ::ttsl::SmallVector<int64_t>{2, 3, 0, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ttnn::deallocate(v11, false);
  ::ttnn::Tensor v13 = ttnn::to_layout(v12, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v12, false);
  ::ttnn::Tensor v14 = ttnn::from_device(v13);
  ttnn::deallocate(v13, false);
  ::ttnn::Tensor v15 = ttnn::operations::conv::conv2d::prepare_conv_weights(v14, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{448, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 2048, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, false, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v14, false);
  ::std::vector<::ttnn::Tensor> v16 = util_create_vec(v15);
  return v16;
}

::std::vector<::ttnn::Tensor> forward_const_eval_81(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 256, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_82(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_83(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_84(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 512, 16, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_85(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_86(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 128, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_87(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{128, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_88(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_89(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_90(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{64, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_91(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_92(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_93(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_94(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_95(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{256, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_96(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_97(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_98(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_99(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v2, ::std::vector<int32_t>{1, 1, 2048, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::permute(v4, ::ttsl::SmallVector<int64_t>{0, 1, 3, 2}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::repeat(v5, ::ttnn::Shape({16, 1, 49, 1}));
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::reshape(v6, ::std::vector<int32_t>{1, 1, 784, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::std::vector<::ttnn::Tensor> v8 = util_create_vec(v7);
  return v8;
}

::std::vector<::ttnn::Tensor> forward_const_eval_100(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 256, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_101(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_102(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_103(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_104(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}

::std::vector<::ttnn::Tensor> forward_const_eval_105(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{512, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::ttnn::Layout::TILE, "OIHW", 256, 512, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_106(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ttnn::distributed::MeshDevice* v4 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v5 = ttnn::reshape(v2, ::std::vector<int32_t>{1024, 1, 1, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::to_device(v3, v4, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_layout(v6, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::multiply(v7, v5, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v9 = ttnn::to_layout(v8, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v8, false);
  ::ttnn::Tensor v10 = ttnn::from_device(v9);
  ttnn::deallocate(v9, false);
  ::ttnn::Tensor v11 = ttnn::operations::conv::conv2d::prepare_conv_weights(v10, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, "OIHW", 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, true, 1, v4, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::std::nullopt);
  ttnn::deallocate(v10, false);
  ::std::vector<::ttnn::Tensor> v12 = util_create_vec(v11);
  return v12;
}

::std::vector<::ttnn::Tensor> forward_const_eval_107(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_layout(v2, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::operations::conv::conv2d::prepare_conv_bias(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}, ::ttnn::Layout::TILE, 512, 512, 16, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, v3, ::ttnn::DataType::BFLOAT8_B, ::ttnn::DataType::BFLOAT8_B, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt);
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
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_59;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_60;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_61;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_62;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_63;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_64;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_65;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_66;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_67;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_68;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_69;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_70;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_71;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_72;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_73;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_74;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_75;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_76;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_77;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_78;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_79;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_80;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_81;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_82;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_83;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_84;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_85;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_86;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_87;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_88;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_89;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_90;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_91;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_92;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_93;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_94;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_95;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_96;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_97;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_98;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_99;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_100;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_101;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_102;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_103;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_104;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_105;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_106;
static ::std::vector<::ttnn::Tensor> g_cached_result_forward_const_eval_107;
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
  ::std::vector<::ttnn::Tensor> v165 = util_create_vec(v90);
  ::std::vector<::ttnn::Tensor>* v166 = &g_cached_result_forward_const_eval_0;
  ttnn::constEvalFuncWrapper(v164, v165, v166);
  ::std::vector<::ttnn::Tensor> v167 = g_cached_result_forward_const_eval_0;
  ::ttnn::Tensor v168 = v167[0];
  ttnn::deallocate(v90, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v169 = &forward_const_eval_1;
  ::std::vector<::ttnn::Tensor> v170 = util_create_vec(v65, v140);
  ::std::vector<::ttnn::Tensor>* v171 = &g_cached_result_forward_const_eval_1;
  ttnn::constEvalFuncWrapper(v169, v170, v171);
  ::std::vector<::ttnn::Tensor> v172 = g_cached_result_forward_const_eval_1;
  ::ttnn::Tensor v173 = v172[0];
  ttnn::deallocate(v140, false);
  ttnn::deallocate(v65, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v174 = &forward_const_eval_2;
  ::std::vector<::ttnn::Tensor> v175 = util_create_vec(v53, v134);
  ::std::vector<::ttnn::Tensor>* v176 = &g_cached_result_forward_const_eval_2;
  ttnn::constEvalFuncWrapper(v174, v175, v176);
  ::std::vector<::ttnn::Tensor> v177 = g_cached_result_forward_const_eval_2;
  ::ttnn::Tensor v178 = v177[0];
  ttnn::deallocate(v134, false);
  ttnn::deallocate(v53, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v179 = &forward_const_eval_3;
  ::std::vector<::ttnn::Tensor> v180 = util_create_vec(v163);
  ::std::vector<::ttnn::Tensor>* v181 = &g_cached_result_forward_const_eval_3;
  ttnn::constEvalFuncWrapper(v179, v180, v181);
  ::std::vector<::ttnn::Tensor> v182 = g_cached_result_forward_const_eval_3;
  ::ttnn::Tensor v183 = v182[0];
  ttnn::deallocate(v163, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v184 = &forward_const_eval_4;
  ::std::vector<::ttnn::Tensor> v185 = util_create_vec(v63, v139);
  ::std::vector<::ttnn::Tensor>* v186 = &g_cached_result_forward_const_eval_4;
  ttnn::constEvalFuncWrapper(v184, v185, v186);
  ::std::vector<::ttnn::Tensor> v187 = g_cached_result_forward_const_eval_4;
  ::ttnn::Tensor v188 = v187[0];
  ttnn::deallocate(v139, false);
  ttnn::deallocate(v63, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v189 = &forward_const_eval_5;
  ::std::vector<::ttnn::Tensor> v190 = util_create_vec(v13, v114);
  ::std::vector<::ttnn::Tensor>* v191 = &g_cached_result_forward_const_eval_5;
  ttnn::constEvalFuncWrapper(v189, v190, v191);
  ::std::vector<::ttnn::Tensor> v192 = g_cached_result_forward_const_eval_5;
  ::ttnn::Tensor v193 = v192[0];
  ttnn::deallocate(v114, false);
  ttnn::deallocate(v13, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v194 = &forward_const_eval_6;
  ::std::vector<::ttnn::Tensor> v195 = util_create_vec(v6);
  ::std::vector<::ttnn::Tensor>* v196 = &g_cached_result_forward_const_eval_6;
  ttnn::constEvalFuncWrapper(v194, v195, v196);
  ::std::vector<::ttnn::Tensor> v197 = g_cached_result_forward_const_eval_6;
  ::ttnn::Tensor v198 = v197[0];
  ttnn::deallocate(v6, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v199 = &forward_const_eval_7;
  ::std::vector<::ttnn::Tensor> v200 = util_create_vec(v82);
  ::std::vector<::ttnn::Tensor>* v201 = &g_cached_result_forward_const_eval_7;
  ttnn::constEvalFuncWrapper(v199, v200, v201);
  ::std::vector<::ttnn::Tensor> v202 = g_cached_result_forward_const_eval_7;
  ::ttnn::Tensor v203 = v202[0];
  ttnn::deallocate(v82, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v204 = &forward_const_eval_8;
  ::std::vector<::ttnn::Tensor> v205 = util_create_vec(v49, v132);
  ::std::vector<::ttnn::Tensor>* v206 = &g_cached_result_forward_const_eval_8;
  ttnn::constEvalFuncWrapper(v204, v205, v206);
  ::std::vector<::ttnn::Tensor> v207 = g_cached_result_forward_const_eval_8;
  ::ttnn::Tensor v208 = v207[0];
  ttnn::deallocate(v132, false);
  ttnn::deallocate(v49, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v209 = &forward_const_eval_9;
  ::std::vector<::ttnn::Tensor> v210 = util_create_vec(v24);
  ::std::vector<::ttnn::Tensor>* v211 = &g_cached_result_forward_const_eval_9;
  ttnn::constEvalFuncWrapper(v209, v210, v211);
  ::std::vector<::ttnn::Tensor> v212 = g_cached_result_forward_const_eval_9;
  ::ttnn::Tensor v213 = v212[0];
  ttnn::deallocate(v24, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v214 = &forward_const_eval_10;
  ::std::vector<::ttnn::Tensor> v215 = util_create_vec(v5, v110);
  ::std::vector<::ttnn::Tensor>* v216 = &g_cached_result_forward_const_eval_10;
  ttnn::constEvalFuncWrapper(v214, v215, v216);
  ::std::vector<::ttnn::Tensor> v217 = g_cached_result_forward_const_eval_10;
  ::ttnn::Tensor v218 = v217[0];
  ttnn::deallocate(v110, false);
  ttnn::deallocate(v5, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v219 = &forward_const_eval_11;
  ::std::vector<::ttnn::Tensor> v220 = util_create_vec(v105, v160);
  ::std::vector<::ttnn::Tensor>* v221 = &g_cached_result_forward_const_eval_11;
  ttnn::constEvalFuncWrapper(v219, v220, v221);
  ::std::vector<::ttnn::Tensor> v222 = g_cached_result_forward_const_eval_11;
  ::ttnn::Tensor v223 = v222[0];
  ttnn::deallocate(v160, false);
  ttnn::deallocate(v105, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v224 = &forward_const_eval_12;
  ::std::vector<::ttnn::Tensor> v225 = util_create_vec(v101, v158);
  ::std::vector<::ttnn::Tensor>* v226 = &g_cached_result_forward_const_eval_12;
  ttnn::constEvalFuncWrapper(v224, v225, v226);
  ::std::vector<::ttnn::Tensor> v227 = g_cached_result_forward_const_eval_12;
  ::ttnn::Tensor v228 = v227[0];
  ttnn::deallocate(v158, false);
  ttnn::deallocate(v101, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v229 = &forward_const_eval_13;
  ::std::vector<::ttnn::Tensor> v230 = util_create_vec(v67, v141);
  ::std::vector<::ttnn::Tensor>* v231 = &g_cached_result_forward_const_eval_13;
  ttnn::constEvalFuncWrapper(v229, v230, v231);
  ::std::vector<::ttnn::Tensor> v232 = g_cached_result_forward_const_eval_13;
  ::ttnn::Tensor v233 = v232[0];
  ttnn::deallocate(v141, false);
  ttnn::deallocate(v67, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v234 = &forward_const_eval_14;
  ::std::vector<::ttnn::Tensor> v235 = util_create_vec(v59, v137);
  ::std::vector<::ttnn::Tensor>* v236 = &g_cached_result_forward_const_eval_14;
  ttnn::constEvalFuncWrapper(v234, v235, v236);
  ::std::vector<::ttnn::Tensor> v237 = g_cached_result_forward_const_eval_14;
  ::ttnn::Tensor v238 = v237[0];
  ttnn::deallocate(v137, false);
  ttnn::deallocate(v59, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v239 = &forward_const_eval_15;
  ::std::vector<::ttnn::Tensor> v240 = util_create_vec(v7, v111);
  ::std::vector<::ttnn::Tensor>* v241 = &g_cached_result_forward_const_eval_15;
  ttnn::constEvalFuncWrapper(v239, v240, v241);
  ::std::vector<::ttnn::Tensor> v242 = g_cached_result_forward_const_eval_15;
  ::ttnn::Tensor v243 = v242[0];
  ttnn::deallocate(v111, false);
  ttnn::deallocate(v7, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v244 = &forward_const_eval_16;
  ::std::vector<::ttnn::Tensor> v245 = util_create_vec(v91, v153);
  ::std::vector<::ttnn::Tensor>* v246 = &g_cached_result_forward_const_eval_16;
  ttnn::constEvalFuncWrapper(v244, v245, v246);
  ::std::vector<::ttnn::Tensor> v247 = g_cached_result_forward_const_eval_16;
  ::ttnn::Tensor v248 = v247[0];
  ttnn::deallocate(v153, false);
  ttnn::deallocate(v91, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v249 = &forward_const_eval_17;
  ::std::vector<::ttnn::Tensor> v250 = util_create_vec(v35, v125);
  ::std::vector<::ttnn::Tensor>* v251 = &g_cached_result_forward_const_eval_17;
  ttnn::constEvalFuncWrapper(v249, v250, v251);
  ::std::vector<::ttnn::Tensor> v252 = g_cached_result_forward_const_eval_17;
  ::ttnn::Tensor v253 = v252[0];
  ttnn::deallocate(v125, false);
  ttnn::deallocate(v35, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v254 = &forward_const_eval_18;
  ::std::vector<::ttnn::Tensor> v255 = util_create_vec(v8);
  ::std::vector<::ttnn::Tensor>* v256 = &g_cached_result_forward_const_eval_18;
  ttnn::constEvalFuncWrapper(v254, v255, v256);
  ::std::vector<::ttnn::Tensor> v257 = g_cached_result_forward_const_eval_18;
  ::ttnn::Tensor v258 = v257[0];
  ttnn::deallocate(v8, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v259 = &forward_const_eval_19;
  ::std::vector<::ttnn::Tensor> v260 = util_create_vec(v76);
  ::std::vector<::ttnn::Tensor>* v261 = &g_cached_result_forward_const_eval_19;
  ttnn::constEvalFuncWrapper(v259, v260, v261);
  ::std::vector<::ttnn::Tensor> v262 = g_cached_result_forward_const_eval_19;
  ::ttnn::Tensor v263 = v262[0];
  ttnn::deallocate(v76, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v264 = &forward_const_eval_20;
  ::std::vector<::ttnn::Tensor> v265 = util_create_vec(v4);
  ::std::vector<::ttnn::Tensor>* v266 = &g_cached_result_forward_const_eval_20;
  ttnn::constEvalFuncWrapper(v264, v265, v266);
  ::std::vector<::ttnn::Tensor> v267 = g_cached_result_forward_const_eval_20;
  ::ttnn::Tensor v268 = v267[0];
  ttnn::deallocate(v4, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v269 = &forward_const_eval_21;
  ::std::vector<::ttnn::Tensor> v270 = util_create_vec(v77, v146);
  ::std::vector<::ttnn::Tensor>* v271 = &g_cached_result_forward_const_eval_21;
  ttnn::constEvalFuncWrapper(v269, v270, v271);
  ::std::vector<::ttnn::Tensor> v272 = g_cached_result_forward_const_eval_21;
  ::ttnn::Tensor v273 = v272[0];
  ttnn::deallocate(v146, false);
  ttnn::deallocate(v77, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v274 = &forward_const_eval_22;
  ::std::vector<::ttnn::Tensor> v275 = util_create_vec(v47, v131);
  ::std::vector<::ttnn::Tensor>* v276 = &g_cached_result_forward_const_eval_22;
  ttnn::constEvalFuncWrapper(v274, v275, v276);
  ::std::vector<::ttnn::Tensor> v277 = g_cached_result_forward_const_eval_22;
  ::ttnn::Tensor v278 = v277[0];
  ttnn::deallocate(v131, false);
  ttnn::deallocate(v47, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v279 = &forward_const_eval_23;
  ::std::vector<::ttnn::Tensor> v280 = util_create_vec(v87, v151);
  ::std::vector<::ttnn::Tensor>* v281 = &g_cached_result_forward_const_eval_23;
  ttnn::constEvalFuncWrapper(v279, v280, v281);
  ::std::vector<::ttnn::Tensor> v282 = g_cached_result_forward_const_eval_23;
  ::ttnn::Tensor v283 = v282[0];
  ttnn::deallocate(v151, false);
  ttnn::deallocate(v87, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v284 = &forward_const_eval_24;
  ::std::vector<::ttnn::Tensor> v285 = util_create_vec(v78);
  ::std::vector<::ttnn::Tensor>* v286 = &g_cached_result_forward_const_eval_24;
  ttnn::constEvalFuncWrapper(v284, v285, v286);
  ::std::vector<::ttnn::Tensor> v287 = g_cached_result_forward_const_eval_24;
  ::ttnn::Tensor v288 = v287[0];
  ttnn::deallocate(v78, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v289 = &forward_const_eval_25;
  ::std::vector<::ttnn::Tensor> v290 = util_create_vec(v29, v122);
  ::std::vector<::ttnn::Tensor>* v291 = &g_cached_result_forward_const_eval_25;
  ttnn::constEvalFuncWrapper(v289, v290, v291);
  ::std::vector<::ttnn::Tensor> v292 = g_cached_result_forward_const_eval_25;
  ::ttnn::Tensor v293 = v292[0];
  ttnn::deallocate(v122, false);
  ttnn::deallocate(v29, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v294 = &forward_const_eval_26;
  ::std::vector<::ttnn::Tensor> v295 = util_create_vec(v23, v119);
  ::std::vector<::ttnn::Tensor>* v296 = &g_cached_result_forward_const_eval_26;
  ttnn::constEvalFuncWrapper(v294, v295, v296);
  ::std::vector<::ttnn::Tensor> v297 = g_cached_result_forward_const_eval_26;
  ::ttnn::Tensor v298 = v297[0];
  ttnn::deallocate(v119, false);
  ttnn::deallocate(v23, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v299 = &forward_const_eval_27;
  ::std::vector<::ttnn::Tensor> v300 = util_create_vec(v85, v150);
  ::std::vector<::ttnn::Tensor>* v301 = &g_cached_result_forward_const_eval_27;
  ttnn::constEvalFuncWrapper(v299, v300, v301);
  ::std::vector<::ttnn::Tensor> v302 = g_cached_result_forward_const_eval_27;
  ::ttnn::Tensor v303 = v302[0];
  ttnn::deallocate(v150, false);
  ttnn::deallocate(v85, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v304 = &forward_const_eval_28;
  ::std::vector<::ttnn::Tensor> v305 = util_create_vec(v32);
  ::std::vector<::ttnn::Tensor>* v306 = &g_cached_result_forward_const_eval_28;
  ttnn::constEvalFuncWrapper(v304, v305, v306);
  ::std::vector<::ttnn::Tensor> v307 = g_cached_result_forward_const_eval_28;
  ::ttnn::Tensor v308 = v307[0];
  ttnn::deallocate(v32, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v309 = &forward_const_eval_29;
  ::std::vector<::ttnn::Tensor> v310 = util_create_vec(v10);
  ::std::vector<::ttnn::Tensor>* v311 = &g_cached_result_forward_const_eval_29;
  ttnn::constEvalFuncWrapper(v309, v310, v311);
  ::std::vector<::ttnn::Tensor> v312 = g_cached_result_forward_const_eval_29;
  ::ttnn::Tensor v313 = v312[0];
  ttnn::deallocate(v10, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v314 = &forward_const_eval_30;
  ::std::vector<::ttnn::Tensor> v315 = util_create_vec(v107);
  ::std::vector<::ttnn::Tensor>* v316 = &g_cached_result_forward_const_eval_30;
  ttnn::constEvalFuncWrapper(v314, v315, v316);
  ::std::vector<::ttnn::Tensor> v317 = g_cached_result_forward_const_eval_30;
  ::ttnn::Tensor v318 = v317[0];
  ttnn::deallocate(v107, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v319 = &forward_const_eval_31;
  ::std::vector<::ttnn::Tensor> v320 = util_create_vec(v34);
  ::std::vector<::ttnn::Tensor>* v321 = &g_cached_result_forward_const_eval_31;
  ttnn::constEvalFuncWrapper(v319, v320, v321);
  ::std::vector<::ttnn::Tensor> v322 = g_cached_result_forward_const_eval_31;
  ::ttnn::Tensor v323 = v322[0];
  ttnn::deallocate(v34, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v324 = &forward_const_eval_32;
  ::std::vector<::ttnn::Tensor> v325 = util_create_vec(v73, v144);
  ::std::vector<::ttnn::Tensor>* v326 = &g_cached_result_forward_const_eval_32;
  ttnn::constEvalFuncWrapper(v324, v325, v326);
  ::std::vector<::ttnn::Tensor> v327 = g_cached_result_forward_const_eval_32;
  ::ttnn::Tensor v328 = v327[0];
  ttnn::deallocate(v144, false);
  ttnn::deallocate(v73, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v329 = &forward_const_eval_33;
  ::std::vector<::ttnn::Tensor> v330 = util_create_vec(v26);
  ::std::vector<::ttnn::Tensor>* v331 = &g_cached_result_forward_const_eval_33;
  ttnn::constEvalFuncWrapper(v329, v330, v331);
  ::std::vector<::ttnn::Tensor> v332 = g_cached_result_forward_const_eval_33;
  ::ttnn::Tensor v333 = v332[0];
  ttnn::deallocate(v26, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v334 = &forward_const_eval_34;
  ::std::vector<::ttnn::Tensor> v335 = util_create_vec(v79, v147);
  ::std::vector<::ttnn::Tensor>* v336 = &g_cached_result_forward_const_eval_34;
  ttnn::constEvalFuncWrapper(v334, v335, v336);
  ::std::vector<::ttnn::Tensor> v337 = g_cached_result_forward_const_eval_34;
  ::ttnn::Tensor v338 = v337[0];
  ttnn::deallocate(v147, false);
  ttnn::deallocate(v79, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v339 = &forward_const_eval_35;
  ::std::vector<::ttnn::Tensor> v340 = util_create_vec(v39, v127);
  ::std::vector<::ttnn::Tensor>* v341 = &g_cached_result_forward_const_eval_35;
  ttnn::constEvalFuncWrapper(v339, v340, v341);
  ::std::vector<::ttnn::Tensor> v342 = g_cached_result_forward_const_eval_35;
  ::ttnn::Tensor v343 = v342[0];
  ttnn::deallocate(v127, false);
  ttnn::deallocate(v39, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v344 = &forward_const_eval_36;
  ::std::vector<::ttnn::Tensor> v345 = util_create_vec(v17, v116);
  ::std::vector<::ttnn::Tensor>* v346 = &g_cached_result_forward_const_eval_36;
  ttnn::constEvalFuncWrapper(v344, v345, v346);
  ::std::vector<::ttnn::Tensor> v347 = g_cached_result_forward_const_eval_36;
  ::ttnn::Tensor v348 = v347[0];
  ttnn::deallocate(v116, false);
  ttnn::deallocate(v17, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v349 = &forward_const_eval_37;
  ::std::vector<::ttnn::Tensor> v350 = util_create_vec(v66);
  ::std::vector<::ttnn::Tensor>* v351 = &g_cached_result_forward_const_eval_37;
  ttnn::constEvalFuncWrapper(v349, v350, v351);
  ::std::vector<::ttnn::Tensor> v352 = g_cached_result_forward_const_eval_37;
  ::ttnn::Tensor v353 = v352[0];
  ttnn::deallocate(v66, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v354 = &forward_const_eval_38;
  ::std::vector<::ttnn::Tensor> v355 = util_create_vec(v93, v154);
  ::std::vector<::ttnn::Tensor>* v356 = &g_cached_result_forward_const_eval_38;
  ttnn::constEvalFuncWrapper(v354, v355, v356);
  ::std::vector<::ttnn::Tensor> v357 = g_cached_result_forward_const_eval_38;
  ::ttnn::Tensor v358 = v357[0];
  ttnn::deallocate(v154, false);
  ttnn::deallocate(v93, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v359 = &forward_const_eval_39;
  ::std::vector<::ttnn::Tensor> v360 = util_create_vec(v18);
  ::std::vector<::ttnn::Tensor>* v361 = &g_cached_result_forward_const_eval_39;
  ttnn::constEvalFuncWrapper(v359, v360, v361);
  ::std::vector<::ttnn::Tensor> v362 = g_cached_result_forward_const_eval_39;
  ::ttnn::Tensor v363 = v362[0];
  ttnn::deallocate(v18, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v364 = &forward_const_eval_40;
  ::std::vector<::ttnn::Tensor> v365 = util_create_vec(v103, v159);
  ::std::vector<::ttnn::Tensor>* v366 = &g_cached_result_forward_const_eval_40;
  ttnn::constEvalFuncWrapper(v364, v365, v366);
  ::std::vector<::ttnn::Tensor> v367 = g_cached_result_forward_const_eval_40;
  ::ttnn::Tensor v368 = v367[0];
  ttnn::deallocate(v159, false);
  ttnn::deallocate(v103, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v369 = &forward_const_eval_41;
  ::std::vector<::ttnn::Tensor> v370 = util_create_vec(v50);
  ::std::vector<::ttnn::Tensor>* v371 = &g_cached_result_forward_const_eval_41;
  ttnn::constEvalFuncWrapper(v369, v370, v371);
  ::std::vector<::ttnn::Tensor> v372 = g_cached_result_forward_const_eval_41;
  ::ttnn::Tensor v373 = v372[0];
  ttnn::deallocate(v50, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v374 = &forward_const_eval_42;
  ::std::vector<::ttnn::Tensor> v375 = util_create_vec(v14);
  ::std::vector<::ttnn::Tensor>* v376 = &g_cached_result_forward_const_eval_42;
  ttnn::constEvalFuncWrapper(v374, v375, v376);
  ::std::vector<::ttnn::Tensor> v377 = g_cached_result_forward_const_eval_42;
  ::ttnn::Tensor v378 = v377[0];
  ttnn::deallocate(v14, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v379 = &forward_const_eval_43;
  ::std::vector<::ttnn::Tensor> v380 = util_create_vec(v62);
  ::std::vector<::ttnn::Tensor>* v381 = &g_cached_result_forward_const_eval_43;
  ttnn::constEvalFuncWrapper(v379, v380, v381);
  ::std::vector<::ttnn::Tensor> v382 = g_cached_result_forward_const_eval_43;
  ::ttnn::Tensor v383 = v382[0];
  ttnn::deallocate(v62, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v384 = &forward_const_eval_44;
  ::std::vector<::ttnn::Tensor> v385 = util_create_vec(v96);
  ::std::vector<::ttnn::Tensor>* v386 = &g_cached_result_forward_const_eval_44;
  ttnn::constEvalFuncWrapper(v384, v385, v386);
  ::std::vector<::ttnn::Tensor> v387 = g_cached_result_forward_const_eval_44;
  ::ttnn::Tensor v388 = v387[0];
  ttnn::deallocate(v96, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v389 = &forward_const_eval_45;
  ::std::vector<::ttnn::Tensor> v390 = util_create_vec(v161);
  ::std::vector<::ttnn::Tensor>* v391 = &g_cached_result_forward_const_eval_45;
  ttnn::constEvalFuncWrapper(v389, v390, v391);
  ::std::vector<::ttnn::Tensor> v392 = g_cached_result_forward_const_eval_45;
  ::ttnn::Tensor v393 = v392[0];
  ttnn::deallocate(v161, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v394 = &forward_const_eval_46;
  ::std::vector<::ttnn::Tensor> v395 = util_create_vec(v44);
  ::std::vector<::ttnn::Tensor>* v396 = &g_cached_result_forward_const_eval_46;
  ttnn::constEvalFuncWrapper(v394, v395, v396);
  ::std::vector<::ttnn::Tensor> v397 = g_cached_result_forward_const_eval_46;
  ::ttnn::Tensor v398 = v397[0];
  ttnn::deallocate(v44, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v399 = &forward_const_eval_47;
  ::std::vector<::ttnn::Tensor> v400 = util_create_vec(v89, v152);
  ::std::vector<::ttnn::Tensor>* v401 = &g_cached_result_forward_const_eval_47;
  ttnn::constEvalFuncWrapper(v399, v400, v401);
  ::std::vector<::ttnn::Tensor> v402 = g_cached_result_forward_const_eval_47;
  ::ttnn::Tensor v403 = v402[0];
  ttnn::deallocate(v152, false);
  ttnn::deallocate(v89, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v404 = &forward_const_eval_48;
  ::std::vector<::ttnn::Tensor> v405 = util_create_vec(v3, v109);
  ::std::vector<::ttnn::Tensor>* v406 = &g_cached_result_forward_const_eval_48;
  ttnn::constEvalFuncWrapper(v404, v405, v406);
  ::std::vector<::ttnn::Tensor> v407 = g_cached_result_forward_const_eval_48;
  ::ttnn::Tensor v408 = v407[0];
  ttnn::deallocate(v109, false);
  ttnn::deallocate(v3, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v409 = &forward_const_eval_49;
  ::std::vector<::ttnn::Tensor> v410 = util_create_vec(v81, v148);
  ::std::vector<::ttnn::Tensor>* v411 = &g_cached_result_forward_const_eval_49;
  ttnn::constEvalFuncWrapper(v409, v410, v411);
  ::std::vector<::ttnn::Tensor> v412 = g_cached_result_forward_const_eval_49;
  ::ttnn::Tensor v413 = v412[0];
  ttnn::deallocate(v148, false);
  ttnn::deallocate(v81, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v414 = &forward_const_eval_50;
  ::std::vector<::ttnn::Tensor> v415 = util_create_vec(v51, v133);
  ::std::vector<::ttnn::Tensor>* v416 = &g_cached_result_forward_const_eval_50;
  ttnn::constEvalFuncWrapper(v414, v415, v416);
  ::std::vector<::ttnn::Tensor> v417 = g_cached_result_forward_const_eval_50;
  ::ttnn::Tensor v418 = v417[0];
  ttnn::deallocate(v133, false);
  ttnn::deallocate(v51, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v419 = &forward_const_eval_51;
  ::std::vector<::ttnn::Tensor> v420 = util_create_vec(v15, v115);
  ::std::vector<::ttnn::Tensor>* v421 = &g_cached_result_forward_const_eval_51;
  ttnn::constEvalFuncWrapper(v419, v420, v421);
  ::std::vector<::ttnn::Tensor> v422 = g_cached_result_forward_const_eval_51;
  ::ttnn::Tensor v423 = v422[0];
  ttnn::deallocate(v115, false);
  ttnn::deallocate(v15, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v424 = &forward_const_eval_52;
  ::std::vector<::ttnn::Tensor> v425 = util_create_vec(v94);
  ::std::vector<::ttnn::Tensor>* v426 = &g_cached_result_forward_const_eval_52;
  ttnn::constEvalFuncWrapper(v424, v425, v426);
  ::std::vector<::ttnn::Tensor> v427 = g_cached_result_forward_const_eval_52;
  ::ttnn::Tensor v428 = v427[0];
  ttnn::deallocate(v94, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v429 = &forward_const_eval_53;
  ::std::vector<::ttnn::Tensor> v430 = util_create_vec(v75, v145);
  ::std::vector<::ttnn::Tensor>* v431 = &g_cached_result_forward_const_eval_53;
  ttnn::constEvalFuncWrapper(v429, v430, v431);
  ::std::vector<::ttnn::Tensor> v432 = g_cached_result_forward_const_eval_53;
  ::ttnn::Tensor v433 = v432[0];
  ttnn::deallocate(v145, false);
  ttnn::deallocate(v75, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v434 = &forward_const_eval_54;
  ::std::vector<::ttnn::Tensor> v435 = util_create_vec(v33, v124);
  ::std::vector<::ttnn::Tensor>* v436 = &g_cached_result_forward_const_eval_54;
  ttnn::constEvalFuncWrapper(v434, v435, v436);
  ::std::vector<::ttnn::Tensor> v437 = g_cached_result_forward_const_eval_54;
  ::ttnn::Tensor v438 = v437[0];
  ttnn::deallocate(v124, false);
  ttnn::deallocate(v33, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v439 = &forward_const_eval_55;
  ::std::vector<::ttnn::Tensor> v440 = util_create_vec(v36);
  ::std::vector<::ttnn::Tensor>* v441 = &g_cached_result_forward_const_eval_55;
  ttnn::constEvalFuncWrapper(v439, v440, v441);
  ::std::vector<::ttnn::Tensor> v442 = g_cached_result_forward_const_eval_55;
  ::ttnn::Tensor v443 = v442[0];
  ttnn::deallocate(v36, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v444 = &forward_const_eval_56;
  ::std::vector<::ttnn::Tensor> v445 = util_create_vec(v60);
  ::std::vector<::ttnn::Tensor>* v446 = &g_cached_result_forward_const_eval_56;
  ttnn::constEvalFuncWrapper(v444, v445, v446);
  ::std::vector<::ttnn::Tensor> v447 = g_cached_result_forward_const_eval_56;
  ::ttnn::Tensor v448 = v447[0];
  ttnn::deallocate(v60, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v449 = &forward_const_eval_57;
  ::std::vector<::ttnn::Tensor> v450 = util_create_vec(v20);
  ::std::vector<::ttnn::Tensor>* v451 = &g_cached_result_forward_const_eval_57;
  ttnn::constEvalFuncWrapper(v449, v450, v451);
  ::std::vector<::ttnn::Tensor> v452 = g_cached_result_forward_const_eval_57;
  ::ttnn::Tensor v453 = v452[0];
  ttnn::deallocate(v20, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v454 = &forward_const_eval_58;
  ::std::vector<::ttnn::Tensor> v455 = util_create_vec(v64);
  ::std::vector<::ttnn::Tensor>* v456 = &g_cached_result_forward_const_eval_58;
  ttnn::constEvalFuncWrapper(v454, v455, v456);
  ::std::vector<::ttnn::Tensor> v457 = g_cached_result_forward_const_eval_58;
  ::ttnn::Tensor v458 = v457[0];
  ttnn::deallocate(v64, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v459 = &forward_const_eval_59;
  ::std::vector<::ttnn::Tensor> v460 = util_create_vec(v28);
  ::std::vector<::ttnn::Tensor>* v461 = &g_cached_result_forward_const_eval_59;
  ttnn::constEvalFuncWrapper(v459, v460, v461);
  ::std::vector<::ttnn::Tensor> v462 = g_cached_result_forward_const_eval_59;
  ::ttnn::Tensor v463 = v462[0];
  ttnn::deallocate(v28, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v464 = &forward_const_eval_60;
  ::std::vector<::ttnn::Tensor> v465 = util_create_vec(v104);
  ::std::vector<::ttnn::Tensor>* v466 = &g_cached_result_forward_const_eval_60;
  ttnn::constEvalFuncWrapper(v464, v465, v466);
  ::std::vector<::ttnn::Tensor> v467 = g_cached_result_forward_const_eval_60;
  ::ttnn::Tensor v468 = v467[0];
  ttnn::deallocate(v104, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v469 = &forward_const_eval_61;
  ::std::vector<::ttnn::Tensor> v470 = util_create_vec(v38);
  ::std::vector<::ttnn::Tensor>* v471 = &g_cached_result_forward_const_eval_61;
  ttnn::constEvalFuncWrapper(v469, v470, v471);
  ::std::vector<::ttnn::Tensor> v472 = g_cached_result_forward_const_eval_61;
  ::ttnn::Tensor v473 = v472[0];
  ttnn::deallocate(v38, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v474 = &forward_const_eval_62;
  ::std::vector<::ttnn::Tensor> v475 = util_create_vec(v46);
  ::std::vector<::ttnn::Tensor>* v476 = &g_cached_result_forward_const_eval_62;
  ttnn::constEvalFuncWrapper(v474, v475, v476);
  ::std::vector<::ttnn::Tensor> v477 = g_cached_result_forward_const_eval_62;
  ::ttnn::Tensor v478 = v477[0];
  ttnn::deallocate(v46, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v479 = &forward_const_eval_63;
  ::std::vector<::ttnn::Tensor> v480 = util_create_vec(v27, v121);
  ::std::vector<::ttnn::Tensor>* v481 = &g_cached_result_forward_const_eval_63;
  ttnn::constEvalFuncWrapper(v479, v480, v481);
  ::std::vector<::ttnn::Tensor> v482 = g_cached_result_forward_const_eval_63;
  ::ttnn::Tensor v483 = v482[0];
  ttnn::deallocate(v121, false);
  ttnn::deallocate(v27, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v484 = &forward_const_eval_64;
  ::std::vector<::ttnn::Tensor> v485 = util_create_vec(v84);
  ::std::vector<::ttnn::Tensor>* v486 = &g_cached_result_forward_const_eval_64;
  ttnn::constEvalFuncWrapper(v484, v485, v486);
  ::std::vector<::ttnn::Tensor> v487 = g_cached_result_forward_const_eval_64;
  ::ttnn::Tensor v488 = v487[0];
  ttnn::deallocate(v84, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v489 = &forward_const_eval_65;
  ::std::vector<::ttnn::Tensor> v490 = util_create_vec(v83, v149);
  ::std::vector<::ttnn::Tensor>* v491 = &g_cached_result_forward_const_eval_65;
  ttnn::constEvalFuncWrapper(v489, v490, v491);
  ::std::vector<::ttnn::Tensor> v492 = g_cached_result_forward_const_eval_65;
  ::ttnn::Tensor v493 = v492[0];
  ttnn::deallocate(v149, false);
  ttnn::deallocate(v83, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v494 = &forward_const_eval_66;
  ::std::vector<::ttnn::Tensor> v495 = util_create_vec(v57, v136);
  ::std::vector<::ttnn::Tensor>* v496 = &g_cached_result_forward_const_eval_66;
  ttnn::constEvalFuncWrapper(v494, v495, v496);
  ::std::vector<::ttnn::Tensor> v497 = g_cached_result_forward_const_eval_66;
  ::ttnn::Tensor v498 = v497[0];
  ttnn::deallocate(v136, false);
  ttnn::deallocate(v57, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v499 = &forward_const_eval_67;
  ::std::vector<::ttnn::Tensor> v500 = util_create_vec(v86);
  ::std::vector<::ttnn::Tensor>* v501 = &g_cached_result_forward_const_eval_67;
  ttnn::constEvalFuncWrapper(v499, v500, v501);
  ::std::vector<::ttnn::Tensor> v502 = g_cached_result_forward_const_eval_67;
  ::ttnn::Tensor v503 = v502[0];
  ttnn::deallocate(v86, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v504 = &forward_const_eval_68;
  ::std::vector<::ttnn::Tensor> v505 = util_create_vec(v19, v117);
  ::std::vector<::ttnn::Tensor>* v506 = &g_cached_result_forward_const_eval_68;
  ttnn::constEvalFuncWrapper(v504, v505, v506);
  ::std::vector<::ttnn::Tensor> v507 = g_cached_result_forward_const_eval_68;
  ::ttnn::Tensor v508 = v507[0];
  ttnn::deallocate(v117, false);
  ttnn::deallocate(v19, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v509 = &forward_const_eval_69;
  ::std::vector<::ttnn::Tensor> v510 = util_create_vec(v98);
  ::std::vector<::ttnn::Tensor>* v511 = &g_cached_result_forward_const_eval_69;
  ttnn::constEvalFuncWrapper(v509, v510, v511);
  ::std::vector<::ttnn::Tensor> v512 = g_cached_result_forward_const_eval_69;
  ::ttnn::Tensor v513 = v512[0];
  ttnn::deallocate(v98, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v514 = &forward_const_eval_70;
  ::std::vector<::ttnn::Tensor> v515 = util_create_vec(v99, v157);
  ::std::vector<::ttnn::Tensor>* v516 = &g_cached_result_forward_const_eval_70;
  ttnn::constEvalFuncWrapper(v514, v515, v516);
  ::std::vector<::ttnn::Tensor> v517 = g_cached_result_forward_const_eval_70;
  ::ttnn::Tensor v518 = v517[0];
  ttnn::deallocate(v157, false);
  ttnn::deallocate(v99, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v519 = &forward_const_eval_71;
  ::std::vector<::ttnn::Tensor> v520 = util_create_vec(v92);
  ::std::vector<::ttnn::Tensor>* v521 = &g_cached_result_forward_const_eval_71;
  ttnn::constEvalFuncWrapper(v519, v520, v521);
  ::std::vector<::ttnn::Tensor> v522 = g_cached_result_forward_const_eval_71;
  ::ttnn::Tensor v523 = v522[0];
  ttnn::deallocate(v92, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v524 = &forward_const_eval_72;
  ::std::vector<::ttnn::Tensor> v525 = util_create_vec(v56);
  ::std::vector<::ttnn::Tensor>* v526 = &g_cached_result_forward_const_eval_72;
  ttnn::constEvalFuncWrapper(v524, v525, v526);
  ::std::vector<::ttnn::Tensor> v527 = g_cached_result_forward_const_eval_72;
  ::ttnn::Tensor v528 = v527[0];
  ttnn::deallocate(v56, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v529 = &forward_const_eval_73;
  ::std::vector<::ttnn::Tensor> v530 = util_create_vec(v97, v156);
  ::std::vector<::ttnn::Tensor>* v531 = &g_cached_result_forward_const_eval_73;
  ttnn::constEvalFuncWrapper(v529, v530, v531);
  ::std::vector<::ttnn::Tensor> v532 = g_cached_result_forward_const_eval_73;
  ::ttnn::Tensor v533 = v532[0];
  ttnn::deallocate(v156, false);
  ttnn::deallocate(v97, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v534 = &forward_const_eval_74;
  ::std::vector<::ttnn::Tensor> v535 = util_create_vec(v58);
  ::std::vector<::ttnn::Tensor>* v536 = &g_cached_result_forward_const_eval_74;
  ttnn::constEvalFuncWrapper(v534, v535, v536);
  ::std::vector<::ttnn::Tensor> v537 = g_cached_result_forward_const_eval_74;
  ::ttnn::Tensor v538 = v537[0];
  ttnn::deallocate(v58, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v539 = &forward_const_eval_75;
  ::std::vector<::ttnn::Tensor> v540 = util_create_vec(v22);
  ::std::vector<::ttnn::Tensor>* v541 = &g_cached_result_forward_const_eval_75;
  ttnn::constEvalFuncWrapper(v539, v540, v541);
  ::std::vector<::ttnn::Tensor> v542 = g_cached_result_forward_const_eval_75;
  ::ttnn::Tensor v543 = v542[0];
  ttnn::deallocate(v22, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v544 = &forward_const_eval_76;
  ::std::vector<::ttnn::Tensor> v545 = util_create_vec(v61, v138);
  ::std::vector<::ttnn::Tensor>* v546 = &g_cached_result_forward_const_eval_76;
  ttnn::constEvalFuncWrapper(v544, v545, v546);
  ::std::vector<::ttnn::Tensor> v547 = g_cached_result_forward_const_eval_76;
  ::ttnn::Tensor v548 = v547[0];
  ttnn::deallocate(v138, false);
  ttnn::deallocate(v61, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v549 = &forward_const_eval_77;
  ::std::vector<::ttnn::Tensor> v550 = util_create_vec(v88);
  ::std::vector<::ttnn::Tensor>* v551 = &g_cached_result_forward_const_eval_77;
  ttnn::constEvalFuncWrapper(v549, v550, v551);
  ::std::vector<::ttnn::Tensor> v552 = g_cached_result_forward_const_eval_77;
  ::ttnn::Tensor v553 = v552[0];
  ttnn::deallocate(v88, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v554 = &forward_const_eval_78;
  ::std::vector<::ttnn::Tensor> v555 = util_create_vec(v102);
  ::std::vector<::ttnn::Tensor>* v556 = &g_cached_result_forward_const_eval_78;
  ttnn::constEvalFuncWrapper(v554, v555, v556);
  ::std::vector<::ttnn::Tensor> v557 = g_cached_result_forward_const_eval_78;
  ::ttnn::Tensor v558 = v557[0];
  ttnn::deallocate(v102, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v559 = &forward_const_eval_79;
  ::std::vector<::ttnn::Tensor> v560 = util_create_vec(v12);
  ::std::vector<::ttnn::Tensor>* v561 = &g_cached_result_forward_const_eval_79;
  ttnn::constEvalFuncWrapper(v559, v560, v561);
  ::std::vector<::ttnn::Tensor> v562 = g_cached_result_forward_const_eval_79;
  ::ttnn::Tensor v563 = v562[0];
  ttnn::deallocate(v12, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v564 = &forward_const_eval_80;
  ::std::vector<::ttnn::Tensor> v565 = util_create_vec(v95, v155);
  ::std::vector<::ttnn::Tensor>* v566 = &g_cached_result_forward_const_eval_80;
  ttnn::constEvalFuncWrapper(v564, v565, v566);
  ::std::vector<::ttnn::Tensor> v567 = g_cached_result_forward_const_eval_80;
  ::ttnn::Tensor v568 = v567[0];
  ttnn::deallocate(v155, false);
  ttnn::deallocate(v95, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v569 = &forward_const_eval_81;
  ::std::vector<::ttnn::Tensor> v570 = util_create_vec(v54);
  ::std::vector<::ttnn::Tensor>* v571 = &g_cached_result_forward_const_eval_81;
  ttnn::constEvalFuncWrapper(v569, v570, v571);
  ::std::vector<::ttnn::Tensor> v572 = g_cached_result_forward_const_eval_81;
  ::ttnn::Tensor v573 = v572[0];
  ttnn::deallocate(v54, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v574 = &forward_const_eval_82;
  ::std::vector<::ttnn::Tensor> v575 = util_create_vec(v68);
  ::std::vector<::ttnn::Tensor>* v576 = &g_cached_result_forward_const_eval_82;
  ttnn::constEvalFuncWrapper(v574, v575, v576);
  ::std::vector<::ttnn::Tensor> v577 = g_cached_result_forward_const_eval_82;
  ::ttnn::Tensor v578 = v577[0];
  ttnn::deallocate(v68, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v579 = &forward_const_eval_83;
  ::std::vector<::ttnn::Tensor> v580 = util_create_vec(v16);
  ::std::vector<::ttnn::Tensor>* v581 = &g_cached_result_forward_const_eval_83;
  ttnn::constEvalFuncWrapper(v579, v580, v581);
  ::std::vector<::ttnn::Tensor> v582 = g_cached_result_forward_const_eval_83;
  ::ttnn::Tensor v583 = v582[0];
  ttnn::deallocate(v16, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v584 = &forward_const_eval_84;
  ::std::vector<::ttnn::Tensor> v585 = util_create_vec(v106);
  ::std::vector<::ttnn::Tensor>* v586 = &g_cached_result_forward_const_eval_84;
  ttnn::constEvalFuncWrapper(v584, v585, v586);
  ::std::vector<::ttnn::Tensor> v587 = g_cached_result_forward_const_eval_84;
  ::ttnn::Tensor v588 = v587[0];
  ttnn::deallocate(v106, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v589 = &forward_const_eval_85;
  ::std::vector<::ttnn::Tensor> v590 = util_create_vec(v45, v130);
  ::std::vector<::ttnn::Tensor>* v591 = &g_cached_result_forward_const_eval_85;
  ttnn::constEvalFuncWrapper(v589, v590, v591);
  ::std::vector<::ttnn::Tensor> v592 = g_cached_result_forward_const_eval_85;
  ::ttnn::Tensor v593 = v592[0];
  ttnn::deallocate(v130, false);
  ttnn::deallocate(v45, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v594 = &forward_const_eval_86;
  ::std::vector<::ttnn::Tensor> v595 = util_create_vec(v25, v120);
  ::std::vector<::ttnn::Tensor>* v596 = &g_cached_result_forward_const_eval_86;
  ttnn::constEvalFuncWrapper(v594, v595, v596);
  ::std::vector<::ttnn::Tensor> v597 = g_cached_result_forward_const_eval_86;
  ::ttnn::Tensor v598 = v597[0];
  ttnn::deallocate(v120, false);
  ttnn::deallocate(v25, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v599 = &forward_const_eval_87;
  ::std::vector<::ttnn::Tensor> v600 = util_create_vec(v41, v128);
  ::std::vector<::ttnn::Tensor>* v601 = &g_cached_result_forward_const_eval_87;
  ttnn::constEvalFuncWrapper(v599, v600, v601);
  ::std::vector<::ttnn::Tensor> v602 = g_cached_result_forward_const_eval_87;
  ::ttnn::Tensor v603 = v602[0];
  ttnn::deallocate(v128, false);
  ttnn::deallocate(v41, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v604 = &forward_const_eval_88;
  ::std::vector<::ttnn::Tensor> v605 = util_create_vec(v9, v112);
  ::std::vector<::ttnn::Tensor>* v606 = &g_cached_result_forward_const_eval_88;
  ttnn::constEvalFuncWrapper(v604, v605, v606);
  ::std::vector<::ttnn::Tensor> v607 = g_cached_result_forward_const_eval_88;
  ::ttnn::Tensor v608 = v607[0];
  ttnn::deallocate(v112, false);
  ttnn::deallocate(v9, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v609 = &forward_const_eval_89;
  ::std::vector<::ttnn::Tensor> v610 = util_create_vec(v11, v113);
  ::std::vector<::ttnn::Tensor>* v611 = &g_cached_result_forward_const_eval_89;
  ttnn::constEvalFuncWrapper(v609, v610, v611);
  ::std::vector<::ttnn::Tensor> v612 = g_cached_result_forward_const_eval_89;
  ::ttnn::Tensor v613 = v612[0];
  ttnn::deallocate(v113, false);
  ttnn::deallocate(v11, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v614 = &forward_const_eval_90;
  ::std::vector<::ttnn::Tensor> v615 = util_create_vec(v21, v118);
  ::std::vector<::ttnn::Tensor>* v616 = &g_cached_result_forward_const_eval_90;
  ttnn::constEvalFuncWrapper(v614, v615, v616);
  ::std::vector<::ttnn::Tensor> v617 = g_cached_result_forward_const_eval_90;
  ::ttnn::Tensor v618 = v617[0];
  ttnn::deallocate(v118, false);
  ttnn::deallocate(v21, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v619 = &forward_const_eval_91;
  ::std::vector<::ttnn::Tensor> v620 = util_create_vec(v40);
  ::std::vector<::ttnn::Tensor>* v621 = &g_cached_result_forward_const_eval_91;
  ttnn::constEvalFuncWrapper(v619, v620, v621);
  ::std::vector<::ttnn::Tensor> v622 = g_cached_result_forward_const_eval_91;
  ::ttnn::Tensor v623 = v622[0];
  ttnn::deallocate(v40, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v624 = &forward_const_eval_92;
  ::std::vector<::ttnn::Tensor> v625 = util_create_vec(v70);
  ::std::vector<::ttnn::Tensor>* v626 = &g_cached_result_forward_const_eval_92;
  ttnn::constEvalFuncWrapper(v624, v625, v626);
  ::std::vector<::ttnn::Tensor> v627 = g_cached_result_forward_const_eval_92;
  ::ttnn::Tensor v628 = v627[0];
  ttnn::deallocate(v70, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v629 = &forward_const_eval_93;
  ::std::vector<::ttnn::Tensor> v630 = util_create_vec(v37, v126);
  ::std::vector<::ttnn::Tensor>* v631 = &g_cached_result_forward_const_eval_93;
  ttnn::constEvalFuncWrapper(v629, v630, v631);
  ::std::vector<::ttnn::Tensor> v632 = g_cached_result_forward_const_eval_93;
  ::ttnn::Tensor v633 = v632[0];
  ttnn::deallocate(v126, false);
  ttnn::deallocate(v37, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v634 = &forward_const_eval_94;
  ::std::vector<::ttnn::Tensor> v635 = util_create_vec(v43, v129);
  ::std::vector<::ttnn::Tensor>* v636 = &g_cached_result_forward_const_eval_94;
  ttnn::constEvalFuncWrapper(v634, v635, v636);
  ::std::vector<::ttnn::Tensor> v637 = g_cached_result_forward_const_eval_94;
  ::ttnn::Tensor v638 = v637[0];
  ttnn::deallocate(v129, false);
  ttnn::deallocate(v43, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v639 = &forward_const_eval_95;
  ::std::vector<::ttnn::Tensor> v640 = util_create_vec(v71, v143);
  ::std::vector<::ttnn::Tensor>* v641 = &g_cached_result_forward_const_eval_95;
  ttnn::constEvalFuncWrapper(v639, v640, v641);
  ::std::vector<::ttnn::Tensor> v642 = g_cached_result_forward_const_eval_95;
  ::ttnn::Tensor v643 = v642[0];
  ttnn::deallocate(v143, false);
  ttnn::deallocate(v71, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v644 = &forward_const_eval_96;
  ::std::vector<::ttnn::Tensor> v645 = util_create_vec(v30);
  ::std::vector<::ttnn::Tensor>* v646 = &g_cached_result_forward_const_eval_96;
  ttnn::constEvalFuncWrapper(v644, v645, v646);
  ::std::vector<::ttnn::Tensor> v647 = g_cached_result_forward_const_eval_96;
  ::ttnn::Tensor v648 = v647[0];
  ttnn::deallocate(v30, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v649 = &forward_const_eval_97;
  ::std::vector<::ttnn::Tensor> v650 = util_create_vec(v74);
  ::std::vector<::ttnn::Tensor>* v651 = &g_cached_result_forward_const_eval_97;
  ttnn::constEvalFuncWrapper(v649, v650, v651);
  ::std::vector<::ttnn::Tensor> v652 = g_cached_result_forward_const_eval_97;
  ::ttnn::Tensor v653 = v652[0];
  ttnn::deallocate(v74, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v654 = &forward_const_eval_98;
  ::std::vector<::ttnn::Tensor> v655 = util_create_vec(v48);
  ::std::vector<::ttnn::Tensor>* v656 = &g_cached_result_forward_const_eval_98;
  ttnn::constEvalFuncWrapper(v654, v655, v656);
  ::std::vector<::ttnn::Tensor> v657 = g_cached_result_forward_const_eval_98;
  ::ttnn::Tensor v658 = v657[0];
  ttnn::deallocate(v48, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v659 = &forward_const_eval_99;
  ::std::vector<::ttnn::Tensor> v660 = util_create_vec(v108);
  ::std::vector<::ttnn::Tensor>* v661 = &g_cached_result_forward_const_eval_99;
  ttnn::constEvalFuncWrapper(v659, v660, v661);
  ::std::vector<::ttnn::Tensor> v662 = g_cached_result_forward_const_eval_99;
  ::ttnn::Tensor v663 = v662[0];
  ttnn::deallocate(v108, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v664 = &forward_const_eval_100;
  ::std::vector<::ttnn::Tensor> v665 = util_create_vec(v52);
  ::std::vector<::ttnn::Tensor>* v666 = &g_cached_result_forward_const_eval_100;
  ttnn::constEvalFuncWrapper(v664, v665, v666);
  ::std::vector<::ttnn::Tensor> v667 = g_cached_result_forward_const_eval_100;
  ::ttnn::Tensor v668 = v667[0];
  ttnn::deallocate(v52, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v669 = &forward_const_eval_101;
  ::std::vector<::ttnn::Tensor> v670 = util_create_vec(v55, v135);
  ::std::vector<::ttnn::Tensor>* v671 = &g_cached_result_forward_const_eval_101;
  ttnn::constEvalFuncWrapper(v669, v670, v671);
  ::std::vector<::ttnn::Tensor> v672 = g_cached_result_forward_const_eval_101;
  ::ttnn::Tensor v673 = v672[0];
  ttnn::deallocate(v135, false);
  ttnn::deallocate(v55, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v674 = &forward_const_eval_102;
  ::std::vector<::ttnn::Tensor> v675 = util_create_vec(v72);
  ::std::vector<::ttnn::Tensor>* v676 = &g_cached_result_forward_const_eval_102;
  ttnn::constEvalFuncWrapper(v674, v675, v676);
  ::std::vector<::ttnn::Tensor> v677 = g_cached_result_forward_const_eval_102;
  ::ttnn::Tensor v678 = v677[0];
  ttnn::deallocate(v72, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v679 = &forward_const_eval_103;
  ::std::vector<::ttnn::Tensor> v680 = util_create_vec(v42);
  ::std::vector<::ttnn::Tensor>* v681 = &g_cached_result_forward_const_eval_103;
  ttnn::constEvalFuncWrapper(v679, v680, v681);
  ::std::vector<::ttnn::Tensor> v682 = g_cached_result_forward_const_eval_103;
  ::ttnn::Tensor v683 = v682[0];
  ttnn::deallocate(v42, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v684 = &forward_const_eval_104;
  ::std::vector<::ttnn::Tensor> v685 = util_create_vec(v80);
  ::std::vector<::ttnn::Tensor>* v686 = &g_cached_result_forward_const_eval_104;
  ttnn::constEvalFuncWrapper(v684, v685, v686);
  ::std::vector<::ttnn::Tensor> v687 = g_cached_result_forward_const_eval_104;
  ::ttnn::Tensor v688 = v687[0];
  ttnn::deallocate(v80, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v689 = &forward_const_eval_105;
  ::std::vector<::ttnn::Tensor> v690 = util_create_vec(v31, v123);
  ::std::vector<::ttnn::Tensor>* v691 = &g_cached_result_forward_const_eval_105;
  ttnn::constEvalFuncWrapper(v689, v690, v691);
  ::std::vector<::ttnn::Tensor> v692 = g_cached_result_forward_const_eval_105;
  ::ttnn::Tensor v693 = v692[0];
  ttnn::deallocate(v123, false);
  ttnn::deallocate(v31, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v694 = &forward_const_eval_106;
  ::std::vector<::ttnn::Tensor> v695 = util_create_vec(v69, v142);
  ::std::vector<::ttnn::Tensor>* v696 = &g_cached_result_forward_const_eval_106;
  ttnn::constEvalFuncWrapper(v694, v695, v696);
  ::std::vector<::ttnn::Tensor> v697 = g_cached_result_forward_const_eval_106;
  ::ttnn::Tensor v698 = v697[0];
  ttnn::deallocate(v142, false);
  ttnn::deallocate(v69, false);
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)> v699 = &forward_const_eval_107;
  ::std::vector<::ttnn::Tensor> v700 = util_create_vec(v100);
  ::std::vector<::ttnn::Tensor>* v701 = &g_cached_result_forward_const_eval_107;
  ttnn::constEvalFuncWrapper(v699, v700, v701);
  ::std::vector<::ttnn::Tensor> v702 = g_cached_result_forward_const_eval_107;
  ::ttnn::Tensor v703 = v702[0];
  ttnn::deallocate(v100, false);
  ttnn::distributed::MeshDevice* v704 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v705 = ttnn::permute(v2, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, 0.000000f);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v706 = ttnn::reshape(v705, ::std::vector<int32_t>{1, 1, 802816, 3}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v705, false);
  ::ttnn::Tensor v707 = ::std::get<0>(ttnn::conv2d(v706, v408, v704, 3, 64, 16, 224, 224, ::std::array<uint32_t, 2>{7, 7}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{3, 3, 3, 3}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v268, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v706, false);
  ttnn::deallocate(v408, false);
  ttnn::deallocate(v268, false);
  ::ttnn::Tensor v708 = ttnn::to_layout(v707, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v707, false);
  ::ttnn::Tensor v709 = ttnn::max_pool2d(v708, 16, 112, 112, 64, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}, ::std::nullopt, false);
  ttnn::deallocate(v708, false);
  ::ttnn::Tensor v710 = ttnn::to_layout(v709, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v711 = ttnn::to_memory_config(v710, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v710, false);
  ::ttnn::Tensor v712 = ::std::get<0>(ttnn::conv2d(v711, v613, v704, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v563, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v711, false);
  ttnn::deallocate(v613, false);
  ttnn::deallocate(v563, false);
  ::ttnn::Tensor v713 = ttnn::to_memory_config(v712, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v712, false);
  ::ttnn::Tensor v714 = ttnn::to_layout(v709, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v709, false);
  ::ttnn::Tensor v715 = ttnn::to_memory_config(v714, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v714, false);
  ::ttnn::Tensor v716 = ::std::get<0>(ttnn::conv2d(v715, v218, v704, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v198, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v715, false);
  ttnn::deallocate(v218, false);
  ttnn::deallocate(v198, false);
  ::ttnn::Tensor v717 = ::std::get<0>(ttnn::conv2d(v716, v243, v704, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v258, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v716, false);
  ttnn::deallocate(v258, false);
  ttnn::deallocate(v243, false);
  ::ttnn::Tensor v718 = ::std::get<0>(ttnn::conv2d(v717, v608, v704, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v313, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v717, false);
  ttnn::deallocate(v608, false);
  ttnn::deallocate(v313, false);
  ::ttnn::Tensor v719 = ttnn::to_memory_config(v718, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v718, false);
  ::ttnn::Tensor v720 = ttnn::add(v719, v713, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v719, false);
  ttnn::deallocate(v713, false);
  ::ttnn::Tensor v721 = ttnn::relu(v720, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v720, false);
  ::ttnn::Tensor v722 = ttnn::to_memory_config(v721, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v723 = ::std::get<0>(ttnn::conv2d(v722, v193, v704, 256, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v378, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v722, false);
  ttnn::deallocate(v378, false);
  ttnn::deallocate(v193, false);
  ::ttnn::Tensor v724 = ::std::get<0>(ttnn::conv2d(v723, v423, v704, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v583, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v723, false);
  ttnn::deallocate(v583, false);
  ttnn::deallocate(v423, false);
  ::ttnn::Tensor v725 = ::std::get<0>(ttnn::conv2d(v724, v348, v704, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v363, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v724, false);
  ttnn::deallocate(v363, false);
  ttnn::deallocate(v348, false);
  ::ttnn::Tensor v726 = ttnn::to_memory_config(v725, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v725, false);
  ::ttnn::Tensor v727 = ttnn::add(v726, v721, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v726, false);
  ttnn::deallocate(v721, false);
  ::ttnn::Tensor v728 = ttnn::relu(v727, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v727, false);
  ::ttnn::Tensor v729 = ttnn::to_memory_config(v728, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v730 = ::std::get<0>(ttnn::conv2d(v729, v508, v704, 256, 64, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v453, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v729, false);
  ttnn::deallocate(v508, false);
  ttnn::deallocate(v453, false);
  ::ttnn::Tensor v731 = ::std::get<0>(ttnn::conv2d(v730, v618, v704, 64, 64, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v543, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v730, false);
  ttnn::deallocate(v618, false);
  ttnn::deallocate(v543, false);
  ::ttnn::Tensor v732 = ::std::get<0>(ttnn::conv2d(v731, v298, v704, 64, 256, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v213, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v731, false);
  ttnn::deallocate(v298, false);
  ttnn::deallocate(v213, false);
  ::ttnn::Tensor v733 = ttnn::to_memory_config(v732, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v732, false);
  ::ttnn::Tensor v734 = ttnn::add(v733, v728, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v733, false);
  ttnn::deallocate(v728, false);
  ::ttnn::Tensor v735 = ttnn::relu(v734, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v734, false);
  ::ttnn::Tensor v736 = ::std::get<0>(ttnn::conv2d(v735, v693, v704, 256, 512, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v308, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt}));
  ttnn::deallocate(v693, false);
  ttnn::deallocate(v308, false);
  ::ttnn::Tensor v737 = ttnn::to_memory_config(v735, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v735, false);
  ::ttnn::Tensor v738 = ::std::get<0>(ttnn::conv2d(v737, v598, v704, 256, 128, 16, 56, 56, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v333, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}, ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{6, 7}}}}, ::std::array<uint32_t, 2>{800, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v737, false);
  ttnn::deallocate(v598, false);
  ttnn::deallocate(v333, false);
  ::ttnn::Tensor v739 = ttnn::to_memory_config(v738, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{896, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v738, false);
  ::ttnn::Tensor v740 = ::std::get<0>(ttnn::conv2d(v739, v483, v704, 128, 128, 16, 56, 56, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v463, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v739, false);
  ttnn::deallocate(v483, false);
  ttnn::deallocate(v463, false);
  ::ttnn::Tensor v741 = ::std::get<0>(ttnn::conv2d(v740, v293, v704, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v648, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v740, false);
  ttnn::deallocate(v648, false);
  ttnn::deallocate(v293, false);
  ::ttnn::Tensor v742 = ttnn::to_memory_config(v741, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v741, false);
  ::ttnn::Tensor v743 = ttnn::add(v742, v736, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v742, false);
  ttnn::deallocate(v736, false);
  ::ttnn::Tensor v744 = ttnn::relu(v743, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v743, false);
  ::ttnn::Tensor v745 = ttnn::to_memory_config(v744, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v746 = ::std::get<0>(ttnn::conv2d(v745, v438, v704, 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v323, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v745, false);
  ttnn::deallocate(v438, false);
  ttnn::deallocate(v323, false);
  ::ttnn::Tensor v747 = ::std::get<0>(ttnn::conv2d(v746, v253, v704, 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v443, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v746, false);
  ttnn::deallocate(v443, false);
  ttnn::deallocate(v253, false);
  ::ttnn::Tensor v748 = ::std::get<0>(ttnn::conv2d(v747, v633, v704, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v473, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v747, false);
  ttnn::deallocate(v633, false);
  ttnn::deallocate(v473, false);
  ::ttnn::Tensor v749 = ttnn::to_memory_config(v748, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v748, false);
  ::ttnn::Tensor v750 = ttnn::add(v749, v744, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v749, false);
  ttnn::deallocate(v744, false);
  ::ttnn::Tensor v751 = ttnn::relu(v750, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v750, false);
  ::ttnn::Tensor v752 = ttnn::to_memory_config(v751, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v753 = ::std::get<0>(ttnn::conv2d(v752, v343, v704, 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v623, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v752, false);
  ttnn::deallocate(v623, false);
  ttnn::deallocate(v343, false);
  ::ttnn::Tensor v754 = ::std::get<0>(ttnn::conv2d(v753, v603, v704, 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v683, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v753, false);
  ttnn::deallocate(v683, false);
  ttnn::deallocate(v603, false);
  ::ttnn::Tensor v755 = ::std::get<0>(ttnn::conv2d(v754, v638, v704, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v398, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v754, false);
  ttnn::deallocate(v638, false);
  ttnn::deallocate(v398, false);
  ::ttnn::Tensor v756 = ttnn::to_memory_config(v755, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v755, false);
  ::ttnn::Tensor v757 = ttnn::add(v756, v751, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v756, false);
  ttnn::deallocate(v751, false);
  ::ttnn::Tensor v758 = ttnn::relu(v757, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v757, false);
  ::ttnn::Tensor v759 = ttnn::to_memory_config(v758, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v760 = ::std::get<0>(ttnn::conv2d(v759, v593, v704, 512, 128, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v478, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v759, false);
  ttnn::deallocate(v593, false);
  ttnn::deallocate(v478, false);
  ::ttnn::Tensor v761 = ::std::get<0>(ttnn::conv2d(v760, v278, v704, 128, 128, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v658, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v760, false);
  ttnn::deallocate(v658, false);
  ttnn::deallocate(v278, false);
  ::ttnn::Tensor v762 = ::std::get<0>(ttnn::conv2d(v761, v208, v704, 128, 512, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v373, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::HEIGHT_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{224, 512}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v761, false);
  ttnn::deallocate(v373, false);
  ttnn::deallocate(v208, false);
  ::ttnn::Tensor v763 = ttnn::to_memory_config(v762, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v762, false);
  ::ttnn::Tensor v764 = ttnn::add(v763, v758, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v763, false);
  ttnn::deallocate(v758, false);
  ::ttnn::Tensor v765 = ttnn::relu(v764, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v764, false);
  ::ttnn::Tensor v766 = ttnn::to_memory_config(v765, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v767 = ::std::get<0>(ttnn::conv2d(v766, v498, v704, 512, 1024, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v538, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v766, false);
  ttnn::deallocate(v538, false);
  ttnn::deallocate(v498, false);
  ::ttnn::Tensor v768 = ttnn::to_memory_config(v767, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v767, false);
  ::ttnn::Tensor v769 = ttnn::to_memory_config(v765, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v765, false);
  ::ttnn::Tensor v770 = ::std::get<0>(ttnn::conv2d(v769, v418, v704, 512, 256, 16, 28, 28, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v668, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{1568, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v769, false);
  ttnn::deallocate(v668, false);
  ttnn::deallocate(v418, false);
  ::ttnn::Tensor v771 = ::std::get<0>(ttnn::conv2d(v770, v178, v704, 256, 256, 16, 28, 28, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v573, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v770, false);
  ttnn::deallocate(v573, false);
  ttnn::deallocate(v178, false);
  ::ttnn::Tensor v772 = ::std::get<0>(ttnn::conv2d(v771, v673, v704, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v528, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v771, false);
  ttnn::deallocate(v673, false);
  ttnn::deallocate(v528, false);
  ::ttnn::Tensor v773 = ttnn::to_memory_config(v772, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v772, false);
  ::ttnn::Tensor v774 = ttnn::add(v773, v768, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v773, false);
  ttnn::deallocate(v768, false);
  ::ttnn::Tensor v775 = ttnn::relu(v774, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v774, false);
  ::ttnn::Tensor v776 = ttnn::to_memory_config(v775, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v777 = ::std::get<0>(ttnn::conv2d(v776, v238, v704, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v448, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v776, false);
  ttnn::deallocate(v448, false);
  ttnn::deallocate(v238, false);
  ::ttnn::Tensor v778 = ::std::get<0>(ttnn::conv2d(v777, v548, v704, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v383, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v777, false);
  ttnn::deallocate(v548, false);
  ttnn::deallocate(v383, false);
  ::ttnn::Tensor v779 = ::std::get<0>(ttnn::conv2d(v778, v188, v704, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v458, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v778, false);
  ttnn::deallocate(v458, false);
  ttnn::deallocate(v188, false);
  ::ttnn::Tensor v780 = ttnn::to_memory_config(v779, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v779, false);
  ::ttnn::Tensor v781 = ttnn::add(v780, v775, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v780, false);
  ttnn::deallocate(v775, false);
  ::ttnn::Tensor v782 = ttnn::relu(v781, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v781, false);
  ::ttnn::Tensor v783 = ttnn::to_memory_config(v782, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v784 = ::std::get<0>(ttnn::conv2d(v783, v173, v704, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v353, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v783, false);
  ttnn::deallocate(v353, false);
  ttnn::deallocate(v173, false);
  ::ttnn::Tensor v785 = ::std::get<0>(ttnn::conv2d(v784, v233, v704, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v578, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v784, false);
  ttnn::deallocate(v578, false);
  ttnn::deallocate(v233, false);
  ::ttnn::Tensor v786 = ::std::get<0>(ttnn::conv2d(v785, v698, v704, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v628, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v785, false);
  ttnn::deallocate(v698, false);
  ttnn::deallocate(v628, false);
  ::ttnn::Tensor v787 = ttnn::to_memory_config(v786, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v786, false);
  ::ttnn::Tensor v788 = ttnn::add(v787, v782, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v787, false);
  ttnn::deallocate(v782, false);
  ::ttnn::Tensor v789 = ttnn::relu(v788, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v788, false);
  ::ttnn::Tensor v790 = ttnn::to_memory_config(v789, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v791 = ::std::get<0>(ttnn::conv2d(v790, v643, v704, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v678, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v790, false);
  ttnn::deallocate(v678, false);
  ttnn::deallocate(v643, false);
  ::ttnn::Tensor v792 = ::std::get<0>(ttnn::conv2d(v791, v328, v704, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v653, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v791, false);
  ttnn::deallocate(v653, false);
  ttnn::deallocate(v328, false);
  ::ttnn::Tensor v793 = ::std::get<0>(ttnn::conv2d(v792, v433, v704, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v263, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v792, false);
  ttnn::deallocate(v433, false);
  ttnn::deallocate(v263, false);
  ::ttnn::Tensor v794 = ttnn::to_memory_config(v793, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v793, false);
  ::ttnn::Tensor v795 = ttnn::add(v794, v789, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v794, false);
  ttnn::deallocate(v789, false);
  ::ttnn::Tensor v796 = ttnn::relu(v795, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v795, false);
  ::ttnn::Tensor v797 = ttnn::to_memory_config(v796, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v798 = ::std::get<0>(ttnn::conv2d(v797, v273, v704, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v288, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v797, false);
  ttnn::deallocate(v288, false);
  ttnn::deallocate(v273, false);
  ::ttnn::Tensor v799 = ::std::get<0>(ttnn::conv2d(v798, v338, v704, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v688, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v798, false);
  ttnn::deallocate(v688, false);
  ttnn::deallocate(v338, false);
  ::ttnn::Tensor v800 = ::std::get<0>(ttnn::conv2d(v799, v413, v704, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v203, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v799, false);
  ttnn::deallocate(v413, false);
  ttnn::deallocate(v203, false);
  ::ttnn::Tensor v801 = ttnn::to_memory_config(v800, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v800, false);
  ::ttnn::Tensor v802 = ttnn::add(v801, v796, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v801, false);
  ttnn::deallocate(v796, false);
  ::ttnn::Tensor v803 = ttnn::relu(v802, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v802, false);
  ::ttnn::Tensor v804 = ttnn::to_memory_config(v803, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v805 = ::std::get<0>(ttnn::conv2d(v804, v493, v704, 1024, 256, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v488, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v804, false);
  ttnn::deallocate(v493, false);
  ttnn::deallocate(v488, false);
  ::ttnn::Tensor v806 = ::std::get<0>(ttnn::conv2d(v805, v303, v704, 256, 256, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v503, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 32}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v805, false);
  ttnn::deallocate(v503, false);
  ttnn::deallocate(v303, false);
  ::ttnn::Tensor v807 = ::std::get<0>(ttnn::conv2d(v806, v283, v704, 256, 1024, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v553, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v806, false);
  ttnn::deallocate(v553, false);
  ttnn::deallocate(v283, false);
  ::ttnn::Tensor v808 = ttnn::to_memory_config(v807, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v807, false);
  ::ttnn::Tensor v809 = ttnn::add(v808, v803, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v808, false);
  ttnn::deallocate(v803, false);
  ::ttnn::Tensor v810 = ttnn::relu(v809, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v809, false);
  ::ttnn::Tensor v811 = ttnn::to_memory_config(v810, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ::ttnn::Tensor v812 = ::std::get<0>(ttnn::conv2d(v811, v403, v704, 1024, 512, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v168, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 7}}}}, ::std::array<uint32_t, 2>{416, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v811, false);
  ttnn::deallocate(v403, false);
  ttnn::deallocate(v168, false);
  ::ttnn::Tensor v813 = ttnn::to_memory_config(v812, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{448, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v812, false);
  ::ttnn::Tensor v814 = ::std::get<0>(ttnn::conv2d(v813, v248, v704, 512, 512, 16, 14, 14, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v523, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v813, false);
  ttnn::deallocate(v523, false);
  ttnn::deallocate(v248, false);
  ::ttnn::Tensor v815 = ::std::get<0>(ttnn::conv2d(v814, v358, v704, 512, 2048, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v814, false);
  ttnn::deallocate(v358, false);
  ::ttnn::Tensor v816 = ttnn::to_memory_config(v815, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v815, false);
  ::ttnn::Tensor v817 = ttnn::add(v816, v428, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v816, false);
  ttnn::deallocate(v428, false);
  ::ttnn::Tensor v818 = ttnn::to_memory_config(v810, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{448, 128}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}});
  ttnn::deallocate(v810, false);
  ::ttnn::Tensor v819 = ::std::get<0>(ttnn::conv2d(v818, v568, v704, 1024, 2048, 16, 14, 14, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v818, false);
  ttnn::deallocate(v568, false);
  ::ttnn::Tensor v820 = ttnn::to_memory_config(v819, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v819, false);
  ::ttnn::Tensor v821 = ttnn::add(v820, v388, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v820, false);
  ttnn::deallocate(v388, false);
  ::ttnn::Tensor v822 = ttnn::add(v817, v821, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v821, false);
  ttnn::deallocate(v817, false);
  ::ttnn::Tensor v823 = ttnn::relu(v822, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v822, false);
  ::ttnn::Tensor v824 = ::std::get<0>(ttnn::conv2d(v823, v533, v704, 2048, 512, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v513, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v533, false);
  ttnn::deallocate(v513, false);
  ::ttnn::Tensor v825 = ::std::get<0>(ttnn::conv2d(v824, v518, v704, 512, 512, 16, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v703, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v824, false);
  ttnn::deallocate(v703, false);
  ttnn::deallocate(v518, false);
  ::ttnn::Tensor v826 = ::std::get<0>(ttnn::conv2d(v825, v228, v704, 512, 2048, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v825, false);
  ttnn::deallocate(v228, false);
  ::ttnn::Tensor v827 = ttnn::to_memory_config(v826, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v826, false);
  ::ttnn::Tensor v828 = ttnn::add(v827, v558, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v827, false);
  ttnn::deallocate(v558, false);
  ::ttnn::Tensor v829 = ttnn::add(v828, v823, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v828, false);
  ttnn::deallocate(v823, false);
  ::ttnn::Tensor v830 = ttnn::relu(v829, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v829, false);
  ::ttnn::Tensor v831 = ::std::get<0>(ttnn::conv2d(v830, v368, v704, 2048, 512, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v468, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v468, false);
  ttnn::deallocate(v368, false);
  ::ttnn::Tensor v832 = ::std::get<0>(ttnn::conv2d(v831, v223, v704, 512, 512, 16, 7, 7, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, v588, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B, .activation = "relu"}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 64}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v831, false);
  ttnn::deallocate(v588, false);
  ttnn::deallocate(v223, false);
  ::ttnn::Tensor v833 = ::std::get<0>(ttnn::conv2d(v832, v393, v704, 512, 2048, 16, 7, 7, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0}, ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::BFLOAT8_B, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT8_B}, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::BLOCK_SHARDED, ::ttnn::BufferType::L1, ::ttnn::ShardSpec{::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}}}}, ::std::array<uint32_t, 2>{128, 256}, ::ttnn::types::ShardOrientation::ROW_MAJOR, ::ttnn::types::ShardMode::PHYSICAL}}));
  ttnn::deallocate(v832, false);
  ttnn::deallocate(v393, false);
  ::ttnn::Tensor v834 = ttnn::to_memory_config(v833, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v833, false);
  ::ttnn::Tensor v835 = ttnn::multiply(v834, v318, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v834, false);
  ttnn::deallocate(v318, false);
  ::ttnn::Tensor v836 = ttnn::add(v835, v663, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v835, false);
  ttnn::deallocate(v663, false);
  ::ttnn::Tensor v837 = ttnn::add(v836, v830, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v836, false);
  ttnn::deallocate(v830, false);
  ::ttnn::Tensor v838 = ttnn::relu(v837, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v837, false);
  ::ttnn::Tensor v839 = ttnn::reshape(v838, ::std::vector<int32_t>{16, 1, 49, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v838, false);
  ::ttnn::Tensor v840 = ttnn::mean(v839, ::ttsl::SmallVector<int32_t>{-2}, true, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v839, false);
  ::ttnn::Tensor v841 = ttnn::reshape(v840, ::std::vector<int32_t>{16, 2048}, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v840, false);
  ::ttnn::Tensor v842 = ttnn::matmul(v841, v162, false, false, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v841, false);
  ttnn::deallocate(v162, false);
  ::ttnn::Tensor v843 = ttnn::add(v842, v183, ::ttnn::DataType::BFLOAT8_B, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v842, false);
  ttnn::deallocate(v183, false);
  ::std::vector<::ttnn::Tensor> v844 = util_create_vec(v843);
  return v844;
}

::std::vector<::ttnn::Tensor> create_inputs_for_forward() {
  ttnn::distributed::MeshDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({16, 3, 224, 224}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::to_device(v6, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::to_device(v8, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::to_device(v10, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::to_device(v12, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::to_device(v14, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::to_device(v16, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::to_device(v18, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::to_device(v20, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::to_device(v22, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::to_device(v24, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::to_device(v26, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::to_device(v28, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::to_device(v30, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::to_device(v32, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::to_device(v34, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::to_device(v36, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::to_device(v38, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::to_device(v40, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::ones(::ttnn::Shape({1, 1, 1, 64}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::to_device(v42, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::to_device(v44, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::to_device(v46, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::to_device(v48, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::to_device(v50, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::to_device(v52, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::to_device(v54, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v57 = ttnn::to_device(v56, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v58 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v59 = ttnn::to_device(v58, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v60 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v61 = ttnn::to_device(v60, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v62 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v63 = ttnn::to_device(v62, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v64 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v65 = ttnn::to_device(v64, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v66 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v67 = ttnn::to_device(v66, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v68 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v69 = ttnn::to_device(v68, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v70 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v71 = ttnn::to_device(v70, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v72 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v73 = ttnn::to_device(v72, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v74 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v75 = ttnn::to_device(v74, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v76 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v77 = ttnn::to_device(v76, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v78 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v79 = ttnn::to_device(v78, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v80 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v81 = ttnn::to_device(v80, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v82 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v83 = ttnn::to_device(v82, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v84 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v85 = ttnn::to_device(v84, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v86 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v87 = ttnn::to_device(v86, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v88 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v89 = ttnn::to_device(v88, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v90 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v91 = ttnn::to_device(v90, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v92 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v93 = ttnn::to_device(v92, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v94 = ttnn::ones(::ttnn::Shape({1, 1, 1, 128}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v95 = ttnn::to_device(v94, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v96 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v97 = ttnn::to_device(v96, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v98 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v99 = ttnn::to_device(v98, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v100 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v101 = ttnn::to_device(v100, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v102 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v103 = ttnn::to_device(v102, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v104 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v105 = ttnn::to_device(v104, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v106 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v107 = ttnn::to_device(v106, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v108 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v109 = ttnn::to_device(v108, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v110 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v111 = ttnn::to_device(v110, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v112 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v113 = ttnn::to_device(v112, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v114 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v115 = ttnn::to_device(v114, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v116 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v117 = ttnn::to_device(v116, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v118 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v119 = ttnn::to_device(v118, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v120 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v121 = ttnn::to_device(v120, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v122 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v123 = ttnn::to_device(v122, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v124 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v125 = ttnn::to_device(v124, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v126 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v127 = ttnn::to_device(v126, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v128 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v129 = ttnn::to_device(v128, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v130 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v131 = ttnn::to_device(v130, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v132 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v133 = ttnn::to_device(v132, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v134 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v135 = ttnn::to_device(v134, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v136 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v137 = ttnn::to_device(v136, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v138 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v139 = ttnn::to_device(v138, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v140 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v141 = ttnn::to_device(v140, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v142 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v143 = ttnn::to_device(v142, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v144 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v145 = ttnn::to_device(v144, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v146 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v147 = ttnn::to_device(v146, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v148 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v149 = ttnn::to_device(v148, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v150 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v151 = ttnn::to_device(v150, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v152 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v153 = ttnn::to_device(v152, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v154 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v155 = ttnn::to_device(v154, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v156 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v157 = ttnn::to_device(v156, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v158 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v159 = ttnn::to_device(v158, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v160 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v161 = ttnn::to_device(v160, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v162 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v163 = ttnn::to_device(v162, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v164 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v165 = ttnn::to_device(v164, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v166 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v167 = ttnn::to_device(v166, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v168 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v169 = ttnn::to_device(v168, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v170 = ttnn::ones(::ttnn::Shape({1, 1, 1, 256}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v171 = ttnn::to_device(v170, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v172 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v173 = ttnn::to_device(v172, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v174 = ttnn::ones(::ttnn::Shape({1, 1, 1, 1024}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v175 = ttnn::to_device(v174, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v176 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v177 = ttnn::to_device(v176, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v178 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v179 = ttnn::to_device(v178, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v180 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v181 = ttnn::to_device(v180, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v182 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v183 = ttnn::to_device(v182, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v184 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v185 = ttnn::to_device(v184, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v186 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v187 = ttnn::to_device(v186, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v188 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v189 = ttnn::to_device(v188, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v190 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v191 = ttnn::to_device(v190, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v192 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v193 = ttnn::to_device(v192, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v194 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v195 = ttnn::to_device(v194, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v196 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v197 = ttnn::to_device(v196, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v198 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v199 = ttnn::to_device(v198, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v200 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v201 = ttnn::to_device(v200, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v202 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v203 = ttnn::to_device(v202, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v204 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v205 = ttnn::to_device(v204, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v206 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v207 = ttnn::to_device(v206, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v208 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v209 = ttnn::to_device(v208, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v210 = ttnn::ones(::ttnn::Shape({1, 1, 1, 512}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v211 = ttnn::to_device(v210, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v212 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v213 = ttnn::to_device(v212, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v214 = ttnn::ones(::ttnn::Shape({1, 2048, 1, 1}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
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
  ::ttnn::Tensor v269 = ttnn::ones(::ttnn::Shape({2048, 1000}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v270 = ttnn::to_device(v269, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v271 = ttnn::ones(::ttnn::Shape({1000}), ::ttnn::DataType::BFLOAT8_B, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v272 = ttnn::to_device(v271, v1, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v273 = util_create_vec(v3, v5, v7, v9, v11, v13, v15, v17, v19, v21, v23, v25, v27, v29, v31, v33, v35, v37, v39, v41, v43, v45, v47, v49, v51, v53, v55, v57, v59, v61, v63, v65, v67, v69, v71, v73, v75, v77, v79, v81, v83, v85, v87, v89, v91, v93, v95, v97, v99, v101, v103, v105, v107, v109, v111, v113, v115, v117, v119, v121, v123, v125, v127, v129, v131, v133, v135, v137, v139, v141, v143, v145, v147, v149, v151, v153, v155, v157, v159, v161, v163, v165, v167, v169, v171, v173, v175, v177, v179, v181, v183, v185, v187, v189, v191, v193, v195, v197, v199, v201, v203, v205, v207, v209, v211, v213, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225, v226, v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237, v238, v239, v240, v241, v242, v243, v244, v245, v246, v247, v248, v249, v250, v251, v252, v253, v254, v255, v256, v257, v258, v259, v260, v261, v262, v263, v264, v265, v266, v267, v268, v270, v272);
  return v273;
}

int32_t main() {

  ::std::vector<::ttnn::Tensor> v1 = create_inputs_for_forward();

  ttnn::MeshDevice *device = ttnn::DeviceGetter::getInstance();
  device->enable_program_cache();

  // ==========================================================================================================================================
  // Timing the ResNet50 forward pass - NanoSeconds
  // ==========================================================================================================================================

  constexpr double SECOND_SCALE = 1000000000.0;
  constexpr double MILLISECOND_SCALE = 1000000.0;
  constexpr double MICROSECOND_SCALE = 1000.0;

  int batch_size = v1[0].padded_shape()[0];
  
  // Warmup run
  std::cout << "Running warmup..." << std::endl;
  std::chrono::high_resolution_clock::time_point warmup_start = std::chrono::high_resolution_clock::now();
  ::std::vector<::ttnn::Tensor> v2 = forward(v1);
  std::chrono::high_resolution_clock::time_point warmup_end = std::chrono::high_resolution_clock::now();
  auto warmup_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(warmup_end - warmup_start);
  std::cout << "Warmup completed in " << warmup_duration.count() << " nanoseconds." << std::endl;
  std::cout << "Batch size, number of samples: " << batch_size << std::endl;
  std::cout << "Total iterations: " << 1 << std::endl;
  std::cout << "Total number of samples: " << batch_size << std::endl;
  std::cout << "Samples per second: " << batch_size / (warmup_duration.count() / SECOND_SCALE) << std::endl;
  std::cout << "\n\n";

  // Additional warmup runs
  int warmup_loop_count = 4;
  double warmup_execution_time = 0.0;
  std::cout << "Running additional warmup..." << std::endl;
  for (int i = 0; i < warmup_loop_count; i++) {
    warmup_start = std::chrono::high_resolution_clock::now();
    forward(v1);
    warmup_end = std::chrono::high_resolution_clock::now();
    warmup_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(warmup_end - warmup_start);
    std::cout << "Iteration " << i + 1 << " completed in " << warmup_duration.count() << " nanoseconds." << std::endl;
    warmup_execution_time += warmup_duration.count();
  }

  warmup_execution_time /= SECOND_SCALE;

  std::cout << "ResNet50 additional warmup forward pass completed in: " << warmup_execution_time << " seconds" << std::endl;
  std::cout << "Batch size, number of samples: " << batch_size << std::endl;
  std::cout << "Total iterations: " << warmup_loop_count << std::endl;
  std::cout << "Total number of samples: " << warmup_loop_count * batch_size << std::endl;
  std::cout << "Samples per second: " << warmup_loop_count * batch_size / warmup_execution_time << std::endl;
  std::cout << "\n\n";


  // Measured run
  std::cout << "Running timed ResNet50 forward pass..." << std::endl;

  int loop_count = 16;
  double execution_time = 0.0;

  for (int i = 0; i < loop_count; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    forward(v1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Iteration " << i + 1 << " completed in " << duration.count() << " nanoseconds." << std::endl;
    execution_time += duration.count();
  }

  execution_time /= SECOND_SCALE;

  std::cout << "ResNet50 forward pass completed in: " << execution_time << " seconds" << std::endl;
  std::cout << "Batch size, number of samples: " << batch_size << std::endl;
  std::cout << "Total iterations: " << loop_count << std::endl;
  std::cout << "Total number of samples: " << loop_count * batch_size << std::endl;
  std::cout << "Samples per second: " << loop_count * batch_size / execution_time << std::endl;
  std::cout << "\n\n";

  return 0;

}


