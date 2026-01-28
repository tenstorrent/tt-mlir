// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
#define TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP

#include "tt-metalium/bfloat16.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv_transpose2d/conv_transpose2d.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/embedding_backward/embedding_backward.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads/nlp_concat_heads.hpp"
#include "ttnn/operations/experimental/transformer/nlp_concat_heads_decode/nlp_concat_heads_decode.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/operations/kv_cache/kv_cache.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/rand/rand.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"
#include "ttnn/operations/transformer/split_query_key_value_and_split_heads/split_query_key_value_and_split_heads.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "workarounds.hpp"

#include <cassert>
#include <cstddef>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

template <typename... T>
std::vector<ttnn::Tensor> util_create_vec(T &&...t) {
  return std::vector<ttnn::Tensor>{std::forward<T>(t)...};
}

namespace ttnn {

// DeviceGetter class
//
// Singleton implementation for Device
//
class DeviceGetter {
public:
  static constexpr std::size_t l1SmallSize = 1 << 15;     // 32kB
  static constexpr std::size_t traceRegionSize = 1 << 20; // 1MB

  static ttnn::MeshDevice *getInstance() {
    // If we have an external device, use it.
    if (externalDevice) {
      assert(!hasOwnedDevice);
      return externalDevice;
    }

    static std::shared_ptr<ttnn::MeshDevice> ownedInstance =
        ::ttnn::MeshDevice::create_unit_mesh(0, l1SmallSize, traceRegionSize);
    hasOwnedDevice = true;
    return ownedInstance.get();
  }

  // Set an external device (we don't own it)
  static void setInstance(ttnn::MeshDevice *newInstance) {
    // We don't want to mix and match owned/external devices.
    assert(!hasOwnedDevice);

    // Store the external device pointer.
    externalDevice = newInstance;
  }

private:
  DeviceGetter() = default;

  DeviceGetter(const DeviceGetter &) = delete;
  DeviceGetter &operator=(const DeviceGetter &) = delete;

  // External device (not owned by us).
  static ttnn::MeshDevice *externalDevice;
  // Flag to track if we've set local ownedInstance or not.
  static bool hasOwnedDevice;
};

inline ttnn::MeshDevice *DeviceGetter::externalDevice = nullptr;
inline bool DeviceGetter::hasOwnedDevice = false;

// Function to be exported from the dylib that can be called to set the
// device--extern to avoid mangling.
extern "C" {
void setDevice(ttnn::MeshDevice *device) { DeviceGetter::setInstance(device); }
}

// Wrapper to abstract const-eval logic out of runtime funcs to keep them
// cleaner.  Invokes constEvalFunc iff outputs is empty.
void constEvalFuncWrapper(
    std::function<std::vector<ttnn::Tensor>(std::vector<ttnn::Tensor>)>
        constEvalFunc,
    const std::vector<ttnn::Tensor> &inputs,
    std::vector<ttnn::Tensor> *outputs) {
  if (outputs->empty()) {
    *outputs = constEvalFunc(inputs);
  }
}

uint32_t getScalarFromTensor(const ttnn::Tensor &tensor) {
  assert(tensor.logical_volume() == 1 && "expected scalar tensor");
  assert(tensor.dtype() == ttnn::DataType::UINT32 && "expected uint32 tensor");

  const ::ttnn::Tensor tensorOnHost = ::ttnn::from_device(tensor);
  const ::tt::tt_metal::HostBuffer buffer =
      ::tt::tt_metal::host_buffer::get_host_buffer(tensorOnHost);
  const auto &buf = buffer.view_as<uint32_t>();
  return *buf.begin();
}

::ttnn::Tensor loadTensor(const std::string &filePath, ttnn::Layout layout,
                          ttnn::DataType dtype, ttnn::MeshDevice *device,
                          ttnn::MemoryConfig memoryConfig) {
  ::ttnn::Tensor loadedTensor =
      ::tt::tt_metal::load_tensor_flatbuffer(filePath);

  assert(loadedTensor.device() == nullptr && "loaded tensor must be on host");

  if (loadedTensor.dtype() != dtype) {
    loadedTensor = ::ttnn::to_dtype(loadedTensor, dtype);
  }

  if (loadedTensor.layout() != layout) {
    loadedTensor = ::ttnn::to_layout(loadedTensor, layout);
  }

  if (device != nullptr) {
    loadedTensor = ::ttnn::to_device(loadedTensor, device, memoryConfig);
  }

  return loadedTensor;
}

} // namespace ttnn

//===----------------------------------------------------------------------===//
// CPU Dylib Manager
//
// Manages loading and calling CPU-hoisted functions from a dynamic library.
// The dylib is compiled from LLVM IR by tt-alchemist and contains functions
// that execute on the CPU (e.g., operations that cannot run on the device).
//===----------------------------------------------------------------------===//

class CPUDylibManager {
public:
  // Get the singleton instance.
  static CPUDylibManager &getInstance() {
    static CPUDylibManager instance;
    return instance;
  }

  // Load the CPU dylib from the given path.
  // Returns true if successful, false otherwise.
  bool load(const std::string &path) {
    if (handle_) {
      std::cerr << "CPUDylibManager: dylib already loaded" << std::endl;
      return false;
    }

    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      std::cerr << "CPUDylibManager: failed to load dylib: " << path
                << "\n  Error: " << dlerror() << std::endl;
      return false;
    }

    dylibPath_ = path;
    std::cout << "CPUDylibManager: loaded dylib from " << path << std::endl;
    return true;
  }

  // Load the CPU dylib from the executable's directory.
  // Looks for "cpu_hoisted.so" next to the executable.
  bool loadFromExecutableDir() {
    // Get the path to the executable
    char exePath[4096];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len == -1) {
      std::cerr << "CPUDylibManager: failed to get executable path"
                << std::endl;
      return false;
    }
    exePath[len] = '\0';

    // Extract directory from path
    std::string exeDir(exePath);
    size_t lastSlash = exeDir.rfind('/');
    if (lastSlash != std::string::npos) {
      exeDir = exeDir.substr(0, lastSlash + 1);
    } else {
      exeDir = "./";
    }

    return load(exeDir + "cpu_hoisted.so");
  }

  // Get a function pointer from the loaded dylib.
  // Returns nullptr if the dylib is not loaded or the symbol is not found.
  template <typename FuncType>
  FuncType *getFunction(const std::string &name) {
    if (!handle_) {
      // Try to load from executable directory if not already loaded
      if (!loadFromExecutableDir()) {
        return nullptr;
      }
    }

    // Check cache first
    auto it = symbolCache_.find(name);
    if (it != symbolCache_.end()) {
      return reinterpret_cast<FuncType *>(it->second);
    }

    // Look up the symbol
    dlerror(); // Clear any existing error
    void *sym = dlsym(handle_, name.c_str());
    const char *error = dlerror();
    if (error) {
      std::cerr << "CPUDylibManager: failed to find symbol '" << name
                << "': " << error << std::endl;
      return nullptr;
    }

    // Cache the symbol
    symbolCache_[name] = sym;
    return reinterpret_cast<FuncType *>(sym);
  }

  // Check if the dylib is loaded.
  bool isLoaded() const { return handle_ != nullptr; }

  // Get the path to the loaded dylib.
  const std::string &getPath() const { return dylibPath_; }

  // Unload the dylib.
  void unload() {
    if (handle_) {
      dlclose(handle_);
      handle_ = nullptr;
      dylibPath_.clear();
      symbolCache_.clear();
      std::cout << "CPUDylibManager: unloaded dylib" << std::endl;
    }
  }

  ~CPUDylibManager() { unload(); }

private:
  CPUDylibManager() = default;
  CPUDylibManager(const CPUDylibManager &) = delete;
  CPUDylibManager &operator=(const CPUDylibManager &) = delete;

  void *handle_ = nullptr;
  std::string dylibPath_;
  std::unordered_map<std::string, void *> symbolCache_;
};

// Helper macro to call a CPU-hoisted function.
// Usage: CPU_CALL(function_name, arg1, arg2, ...)
#define CPU_CALL(func_name, ...)                                               \
  do {                                                                         \
    using FuncType = decltype(func_name);                                      \
    auto *func =                                                               \
        CPUDylibManager::getInstance().getFunction<FuncType>(#func_name);      \
    if (func) {                                                                \
      func(__VA_ARGS__);                                                       \
    } else {                                                                   \
      std::cerr << "Failed to call CPU function: " << #func_name << std::endl; \
    }                                                                          \
  } while (0)

#endif // TOOLS_TTNN_STANDALONE_TTNN_PRECOMPILED_HPP
