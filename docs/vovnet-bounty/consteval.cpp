#include "common.hpp"

std::vector<ttnn::Tensor> const_eval_permute(std::vector<ttnn::Tensor> v1) {
  ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                         ttnn::BufferType::DRAM, std::nullopt});
  ttnn::Tensor v5 = ttnn::to_layout(
      v4, ttnn::Layout::TILE, std::nullopt,
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                         ttnn::BufferType::DRAM, std::nullopt});
  ttnn::deallocate(v4, false);
  ttnn::Tensor v6 = ttnn::permute(
      v5, ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                         ttnn::BufferType::DRAM, std::nullopt},
      0.000000f);
  ttnn::deallocate(v5, false);
  return util_create_vec(v6);
}

std::vector<ttnn::Tensor> const_eval_reshape_scalar(std::vector<ttnn::Tensor> v1) {
  ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice* v3 = ttnn::DeviceGetter::getInstance();
  ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                         ttnn::BufferType::DRAM, std::nullopt});
  ttnn::Tensor v5 = ttnn::to_layout(
      v4, ttnn::Layout::TILE, std::nullopt,
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                         ttnn::BufferType::DRAM, std::nullopt});
  ttnn::deallocate(v4, false);
  ttnn::Tensor v6 = ttnn::reshape(
      v5, std::vector<int32_t>{1, 1, 1, 1},
      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                         ttnn::BufferType::DRAM, std::nullopt});
  ttnn::deallocate(v5, false);
  return util_create_vec(v6);
}

std::vector<ttnn::Tensor> const_eval_passthrough(std::vector<ttnn::Tensor> v1) {
  return util_create_vec(v1[0]);
}

// Cache-aware dispatcher used at every codegen call site.
// The slot index is codegen-assigned and stable across runs; `fn` is one of
// the const_eval_* functions above. First call computes & stores; later calls
// return the stored tensor.
ttnn::Tensor cached_const_eval(
    int slot,
    std::vector<ttnn::Tensor> (*fn)(std::vector<ttnn::Tensor>),
    const ttnn::Tensor& input) {
  auto& cache = g_const_eval_cache[slot];
  if (cache.empty()) {
    cache = fn(util_create_vec(input));
  }
  return cache[0];
}
