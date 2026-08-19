// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_TYPES_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_TYPES_H

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace tt::runtime::ttnn {
using OptionalMeshDeviceRef =
    std::optional<std::reference_wrapper<::ttnn::MeshDevice>>;
using TensorMap = std::unordered_map<uint32_t, ::tt::runtime::Tensor>;
using TensorPtrMap = std::unordered_map<uint32_t, ::tt::runtime::Tensor *>;
using GlobalSemaphoreMap =
    std::unordered_map<uint32_t, ::tt::runtime::GlobalSemaphore>;
using TensorPtrMapIterator = typename TensorPtrMap::iterator;
using GlobalSemaphoreMapIterator = typename GlobalSemaphoreMap::iterator;
class TTNNTensorWrapper;
using OnDestroyTensorCallback = std::function<void(TTNNTensorWrapper *)>;

// Wrapper for ttnn::Tensor that contains
// additional metadata specific to our ttnn runtime
class TTNNTensorWrapper {
public:
  TTNNTensorWrapper(
      const ::ttnn::Tensor &tensor,
      const std::optional<::ttnn::MeshEvent> &meshEvent = std::nullopt,
      bool retain = false)
      : tensor(tensor), meshEvent(meshEvent), retain(retain),
        version(getLatestVersion()) {}

  ~TTNNTensorWrapper() {
    for (const auto &callback : onDestroyCallbacks) {
      callback(this);
    }
  }

  TTNNTensorWrapper(const TTNNTensorWrapper &other) = delete;
  TTNNTensorWrapper &operator=(const TTNNTensorWrapper &other) = delete;
  TTNNTensorWrapper(TTNNTensorWrapper &&other) = delete;
  TTNNTensorWrapper &operator=(TTNNTensorWrapper &&other) = delete;

  const ::ttnn::Tensor &getTensor() const { return tensor; }
  ::ttnn::Tensor &getTensor() { return tensor; }

  const std::optional<::ttnn::MeshEvent> &getMeshEvent() const {
    std::shared_lock<std::shared_mutex> lock(meshEventMutex);
    return meshEvent;
  }
  void setMeshEvent(const ::ttnn::MeshEvent &meshEvent) {
    std::unique_lock<std::shared_mutex> lock(meshEventMutex);
    this->meshEvent = meshEvent;
  }

  void registerOnDestroyCallback(OnDestroyTensorCallback callback) {
    onDestroyCallbacks.push_back(callback);
  }

  bool shouldRetain() const { return retain.load(std::memory_order_relaxed); }
  void setRetain(bool val) { retain.store(val, std::memory_order_relaxed); }

  uint64_t getVersion() const {
    return version.load(std::memory_order_relaxed);
  }
  void updateVersion() {
    version.store(getLatestVersion(), std::memory_order_relaxed);
  }
  void syncVersion(const TTNNTensorWrapper &other) {
    version.store(other.getVersion(), std::memory_order_relaxed);
  }

private:
  ::ttnn::Tensor tensor;

  // The mesh device command queue event associated with this tensor.
  // This is used to synchronize the tensor with the mesh device.
  // Most of the time this will not be set, but it is required now
  // that we are adding non-blocking readbacks and multiple command queue
  // support. This will be used increasingly in the future once we start adding
  // such support to all of our operations.
  mutable std::shared_mutex meshEventMutex;
  std::optional<::ttnn::MeshEvent> meshEvent;

  // Whether the tensor should be retained during execution
  // Setting this to true will prohibit deallocate ops within
  // the program from deallocating the tensor
  std::atomic<bool> retain;
  std::atomic<uint64_t> version;

  static uint64_t getLatestVersion();

  // Callbacks to be called on destruction of the tensor.
  // Used for cleaning up any associated resources with this tensor.
  // E.g., used in consteval caching to remove cache entries associated with an
  // input tensor, once the input tensor is destroyed.
  std::vector<OnDestroyTensorCallback> onDestroyCallbacks;
};

struct LayoutDesc {
  ::ttnn::StorageType storageType;
  ::ttnn::Layout layout;
  ::ttnn::DataType dataType;
  std::optional<::ttnn::MemoryConfig> memoryConfig;

  static std::shared_ptr<LayoutDesc>
  fromTensor(const ::tt::runtime::Tensor &tensor);
  static std::shared_ptr<LayoutDesc>
  fromMemoryDesc(const ::tt::target::ttnn::MemoryDesc *memoryDesc);

  LayoutDesc(const ::ttnn::StorageType &storageType,
             const ::ttnn::Layout &layout, const ::ttnn::DataType &dataType,
             const std::optional<::ttnn::MemoryConfig> &memoryConfig);

  bool isOnHost() const;
  bool isOnDevice() const;
  bool isTilized() const;

  ::flatbuffers::Offset<::tt::target::ttnn::MemoryDesc>
  toMemoryDesc(::flatbuffers::FlatBufferBuilder &fbb) const;

  bool operator==(const LayoutDesc &other) const;
};

class ProgramTensorPool {
public:
  ProgramTensorPool(const std::vector<uint32_t> &programInputIds,
                    const std::vector<uint32_t> &programOutputIds,
                    TensorPtrMap &&liveTensors)
      : programInputIds(programInputIds), programOutputIds(programOutputIds),
        liveTensors(std::move(liveTensors)) {}
  ProgramTensorPool(const ProgramTensorPool &) = delete;
  ProgramTensorPool &operator=(const ProgramTensorPool &) = delete;
  ProgramTensorPool(ProgramTensorPool &&) = default;
  ProgramTensorPool &operator=(ProgramTensorPool &&) = default;

  const ::tt::runtime::Tensor &getRuntimeTensorAndValidate(
      const ::tt::target::ttnn::TensorRef *tensorRef) const;
  ::tt::runtime::Tensor &
  getRuntimeTensorAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef);

  const ::tt::runtime::ttnn::TTNNTensorWrapper &getTTNNTensorWrapperAndValidate(
      const ::tt::target::ttnn::TensorRef *tensorRef) const;
  ::tt::runtime::ttnn::TTNNTensorWrapper &getTTNNTensorWrapperAndValidate(
      const ::tt::target::ttnn::TensorRef *tensorRef);

  const ::ttnn::Tensor &getTTNNTensorAndValidate(
      const ::tt::target::ttnn::TensorRef *tensorRef) const;
  ::ttnn::Tensor &
  getTTNNTensorAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef);

  std::pair<TensorPtrMapIterator, bool>
  insertRuntimeTensorAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef,
                                 const ::tt::runtime::Tensor &runtimeTensor);

  std::pair<TensorPtrMapIterator, bool>
  insertTTNNTensorAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef,
                              const ::ttnn::Tensor &ttnnTensor,
                              bool retain = false);

  std::vector<::tt::runtime::Tensor> gatherOutputTensors();

  TensorPtrMapIterator erase(const ::tt::target::ttnn::TensorRef *tensorRef);

  bool contains(const ::tt::target::ttnn::TensorRef *tensorRef) const {
    return liveTensors.contains(tensorRef->global_id());
  }

  const std::vector<std::uint32_t> &getProgramInputIds() const {
    return programInputIds;
  }

  const std::vector<std::uint32_t> &getProgramOutputIds() const {
    return programOutputIds;
  }

  std::uint32_t getScalarKernelArgAndValidate(
      const ::tt::target::ttnn::TensorRef *tensorRef) const;

private:
  const ::tt::runtime::Tensor &getRuntimeTensor(std::uint32_t globalId) const;
  ::tt::runtime::Tensor &getRuntimeTensor(std::uint32_t globalId);
  std::vector<std::uint32_t> programInputIds;
  std::vector<std::uint32_t> programOutputIds;
  TensorMap intermedTensors;
  TensorPtrMap liveTensors;
};

class ProgramGlobalSemaphorePool {
public:
  ProgramGlobalSemaphorePool(GlobalSemaphoreMap &&liveGlobalSemaphores)
      : liveGlobalSemaphores(std::move(liveGlobalSemaphores)) {}
  ProgramGlobalSemaphorePool(const ProgramGlobalSemaphorePool &) = delete;
  ProgramGlobalSemaphorePool &
  operator=(const ProgramGlobalSemaphorePool &) = delete;
  ProgramGlobalSemaphorePool(ProgramGlobalSemaphorePool &&) = default;
  ProgramGlobalSemaphorePool &
  operator=(ProgramGlobalSemaphorePool &&) = default;

  ::tt::runtime::GlobalSemaphore &getRuntimeGlobalSemaphoreAndValidate(
      const ::tt::target::ttnn::GlobalSemaphoreRef *globalSemaphoreRef);

  ::ttnn::GlobalSemaphore &getTTNNGlobalSemaphoreAndValidate(
      const ::tt::target::ttnn::GlobalSemaphoreRef *globalSemaphoreRef);

  std::pair<GlobalSemaphoreMapIterator, bool>
  insertRuntimeGlobalSemaphoreAndValidate(
      const ::tt::target::ttnn::GlobalSemaphoreRef *globalSemaphoreRef,
      ::tt::runtime::GlobalSemaphore runtimeGlobalSemaphore);

  std::pair<GlobalSemaphoreMapIterator, bool>
  insertTTNNGlobalSemaphoreAndValidate(
      const ::tt::target::ttnn::GlobalSemaphoreRef *globalSemaphoreRef,
      const ::ttnn::GlobalSemaphore &ttnnGlobalSemaphore);

  GlobalSemaphoreMapIterator
  erase(const ::tt::target::ttnn::GlobalSemaphoreRef *globalSemaphoreRef);

  bool contains(
      const ::tt::target::ttnn::GlobalSemaphoreRef *globalSemaphoreRef) const {
    return liveGlobalSemaphores.contains(globalSemaphoreRef->global_id());
  }

private:
  GlobalSemaphoreMap liveGlobalSemaphores;

  ::tt::runtime::GlobalSemaphore &
  getRuntimeGlobalSemaphore(std::uint32_t globalId);
};

class ProgramContext {
public:
  ProgramContext(const std::vector<uint32_t> &programInputIds,
                 const std::vector<uint32_t> &programOutputIds,
                 TensorPtrMap &&liveTensors,
                 GlobalSemaphoreMap &&liveGlobalSemaphores,
                 common::DylibManager &&programDylibManager,
                 ::tt::runtime::Device deviceHandle,
                 const Binary &executableHandle, size_t programIndex = 0)
      : tensorPool(ProgramTensorPool(programInputIds, programOutputIds,
                                     std::move(liveTensors))),
        globalSemaphorePool(
            ProgramGlobalSemaphorePool(std::move(liveGlobalSemaphores))),
        dylibManager(std::move(programDylibManager)),
        deviceHandle(deviceHandle), executableHandle(executableHandle),
        programIndex(programIndex) {
    LOG_ASSERT(deviceHandle.handle, "DeviceHandle cannot be null");
  }

  ProgramContext(const ProgramContext &) = delete;
  ProgramContext &operator=(const ProgramContext &) = delete;
  ProgramContext(ProgramContext &&) = delete;
  ProgramContext &operator=(ProgramContext &&) = delete;

  //
  // Sub Mesh Operations
  //

  const ::tt::runtime::Device &getDeviceHandle() const { return deviceHandle; }
  ::tt::runtime::Device &getDeviceHandle() { return deviceHandle; }

  const ::ttnn::MeshDevice &getMeshDevice() const {
    return deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  }
  ::ttnn::MeshDevice &getMeshDevice() {
    return deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  }
  std::shared_ptr<::ttnn::MeshDevice> getMeshDevicePtr() {
    return deviceHandle.asSharedPtr<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  }

  size_t meshDeviceSize() const { return getMeshDevice().num_devices(); }

  const ::ttnn::MeshShape &meshDeviceShape() const {
    return getMeshDevice().shape();
  }

  //
  // Dylib Manager Operation
  //
  const common::DylibManager &getDylibManager() { return dylibManager; }

  //
  // Tensor Pool Operations
  //
  ProgramTensorPool &getTensorPool() { return tensorPool; }

  const ProgramTensorPool &getTensorPool() const { return tensorPool; }

  //
  // Global Semaphore Pool Operations
  //
  ProgramGlobalSemaphorePool &getGlobalSemaphorePool() {
    return globalSemaphorePool;
  }

  Binary &getExecutableHandle() { return executableHandle; }

  //
  // Program Index getter
  //
  size_t getProgramIndex() const { return programIndex; }

  //
  // Host Scalar Cache Operations
  //
  // Some ops take a hyperparameter as a single-element tensor operand so the
  // graph stays the same from step to step, but call an API that wants a plain
  // float (e.g. ttml::metal::adamw and the AdamW bias-correction terms).
  // Reading one back costs a device-to-host sync, and a training step holds one
  // such op per parameter, all reading the same few scalars. Cache the value
  // for the duration of the program run so the sync happens once per scalar
  // instead of once per op.
  //
  // Keyed by TensorRef global id, which is unique per value in a program: an op
  // that recomputes a scalar produces a new tensor with a new id, and the ops
  // that read scalars back take them as read-only operands, so a cached value
  // cannot go stale within one run.
  std::optional<float> getCachedHostScalar(uint32_t globalId) const {
    auto it = hostScalarCache.find(globalId);
    if (it == hostScalarCache.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void cacheHostScalar(uint32_t globalId, float value) {
    hostScalarCache[globalId] = value;
  }

private:
  ProgramTensorPool tensorPool;

  ProgramGlobalSemaphorePool globalSemaphorePool;

  common::DylibManager dylibManager;

  ::tt::runtime::Device deviceHandle;

  // The executable binary handle
  Binary executableHandle;

  // The index of the program within the binary
  const size_t programIndex;

  // Scalars already read back to host during this program run, by TensorRef
  // global id. See getCachedHostScalar.
  std::unordered_map<uint32_t, float> hostScalarCache;
};

} // namespace tt::runtime::ttnn

#endif
