#include "ttnn-precompiled.hpp"
::ttnn::Tensor alexNet(::ttnn::Tensor v1, ::ttnn::Tensor v2) {
  ttnn::IDevice* v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::reshape(v1, ::std::vector<int32_t>{1, 1, 4608, 192}, ::std::nullopt);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v5 = ttnn::from_device(v4);
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::to_layout(v5, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig {.memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::IDevice *>(nullptr));
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::to_device(v6, v3, ::ttnn::MemoryConfig {.memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v6, false);
  ::std::tuple<::ttnn::Tensor, uint32_t, uint32_t, ::ttnn::Tensor, ::std::optional<::ttnn::Tensor>> v8 = ttnn::conv2d(v7, v2, v3, 192, 384, 32, 12, 12, ::std::array<uint32_t, 2>{3, 3}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 2>{1, 1}, 1, ::std::nullopt, ::ttnn::operations::conv::conv2d::Conv2dConfig{.dtype = ::ttnn::DataType::BFLOAT16, .weights_dtype = ::ttnn::DataType::BFLOAT16}, ::std::nullopt, ::ttnn::MemoryConfig {.memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v9 = ::std::get<0>(v8);
  ttnn::deallocate(v7, false);
  ttnn::deallocate(v2, false);
  ::ttnn::Tensor v10 = ttnn::reshape(v9, ::std::vector<int32_t>{32, 12, 12, 384}, ::std::nullopt);
  ttnn::deallocate(v9, false);
  return v10;
}

std::tuple<::ttnn::Tensor, ::ttnn::Tensor> createInputsFor_alexNet() {
  ttnn::IDevice* v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::ones(::ttnn::Shape({32, 12, 12, 192}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig {.memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v3 = ttnn::to_device(v2, v1, ::ttnn::MemoryConfig {.memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::DRAM});
  ::ttnn::Tensor v4 = ttnn::ones(::ttnn::Shape({384, 192, 3, 3}), ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig {.memory_layout = ::ttnn::TensorMemoryLayout::INTERLEAVED, .buffer_type = ::ttnn::BufferType::SYSTEM_MEMORY});
  return std::make_tuple(v3, v4);
}

int32_t main() {
  ::ttnn::Tensor v1;
  ::ttnn::Tensor v2;
  std::tie(v1, v2) = createInputsFor_alexNet();
  ::ttnn::Tensor v3 = alexNet(v1, v2);
  int32_t v4 = 0;
  return v4;
}


