// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types/global_tensor_cache.h"

#include <tt_stl/indestructible.hpp>

#include "tt/runtime/debug.h"
#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime {

GlobalTensorCache &GlobalTensorCache::getInstance() {
  // We use Indestructible to avoid any static destruction ordering issues. The
  // process is exiting anyway, so we don't care about freeing memory at that
  // point.
  static ttsl::Indestructible<GlobalTensorCache> instance;
  return instance.get();
}

std::string to_string(const CacheKey &key) {
  std::stringstream versions;
  for (const auto &v : key.inputVersions) {
    versions << v << ",";
  }
  return std::to_string(key.deviceId) + ":" + key.funcHash + ":" +
         versions.str();
}

// Get all cached tensors for a function name if the cache is valid
// Returns nullptr if cache is invalid or not found
const std::vector<Tensor> *
GlobalTensorCache::getAll(const CacheKey &key) const {
  std::shared_lock<std::shared_mutex> lock(cacheMutex);
  auto it = cache.find(key);
  if (it == cache.end()) {
    debug::Stats::get().incrementStat("ConstEvalCacheMiss");
    return nullptr;
  }

  debug::Stats::get().incrementStat("ConstEvalCacheHit");

  return &it->second;
}

void GlobalTensorCache::store(
    CacheKey key, std::vector<tt::runtime::Tensor> &inputTensors,
    const std::vector<tt::runtime::Tensor> &outputTensors) {
  std::unique_lock<std::shared_mutex> lock(cacheMutex);
  cache[key] = outputTensors;

  // Register on-destroy callbacks to remove the created cache entry when any of
  // the input tensors are destroyed. No weak_ptr needed since the global cache
  // always exists.
  for (::tt::runtime::Tensor &input : inputTensors) {
    ::tt::runtime::ttnn::TTNNTensor &variant =
        input.as<::tt::runtime::ttnn::TTNNTensor>(DeviceRuntime::TTNN);
    auto *wrapperPtr =
        std::get_if<::tt::runtime::ttnn::TTNNTensorWrapperPtr>(&variant);
    if (!wrapperPtr) {
      LOG_FATAL("Unsupported variant type: global_tensor_cache input wrapper "
                "received a scalar runtime tensor; scalar tensors are only "
                "valid for the KernelArgScalar kernel-arg path");
    }
    ::tt::runtime::ttnn::TTNNTensorWrapper &inputWrapper = **wrapperPtr;
    auto onDestroyCallback =
        [key](::tt::runtime::ttnn::TTNNTensorWrapper *tensor) {
          GlobalTensorCache::getInstance().removeIfExists(key);
        };
    inputWrapper.registerOnDestroyCallback(onDestroyCallback);
  }
}

} // namespace tt::runtime
