// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TYPES_H
#define TT_RUNTIME_DETAIL_TTNN_TYPES_H

#include "tt/runtime/detail/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tt::runtime::ttnn {
using OptionalMeshDeviceRef =
    std::optional<std::reference_wrapper<::ttnn::MeshDevice>>;
using TensorMap = std::unordered_map<uint32_t, ::tt::runtime::Tensor>;
using TensorPtrMap = std::unordered_map<uint32_t, ::tt::runtime::Tensor *>;
using TensorPtrMapIterator = typename TensorPtrMap::iterator;

// Wrapper for ttnn::Tensor that contains
// additional metadata specific to our ttnn runtime
class TTNNTensorWrapper {
public:
  TTNNTensorWrapper(const ::ttnn::Tensor &tensor, bool retain = false)
      : tensor(tensor), retain(retain), version(getLatestVersion()) {}

  TTNNTensorWrapper(const TTNNTensorWrapper &other) = delete;
  TTNNTensorWrapper &operator=(const TTNNTensorWrapper &other) = delete;
  TTNNTensorWrapper(TTNNTensorWrapper &&other) = delete;
  TTNNTensorWrapper &operator=(TTNNTensorWrapper &&other) = delete;

  const ::ttnn::Tensor &getTensor() const { return tensor; }
  ::ttnn::Tensor &getTensor() { return tensor; }

  bool shouldRetain() const { return retain.load(std::memory_order_relaxed); }
  void setRetain(bool val) { retain.store(val, std::memory_order_relaxed); }

  uint64_t getVersion() const {
    return version.load(std::memory_order_relaxed);
  }
  void updateVersion() {
    version.store(getLatestVersion(), std::memory_order_relaxed);
  }

private:
  ::ttnn::Tensor tensor;
  // Whether the tensor should be retained during execution
  // Setting this to true will prohibit deallocate ops within
  // the program from deallocating the tensor
  std::atomic<bool> retain;
  std::atomic<uint64_t> version;

  static std::atomic<uint64_t> getLatestVersion() {
    static std::atomic<uint64_t> latestVersion{0};
    return latestVersion++;
  }
};

struct LayoutDesc {
  ::ttnn::StorageType storageType;
  ::ttnn::Layout layout;
  ::ttnn::DataType dataType;
  std::optional<::ttnn::MemoryConfig> memoryConfig;

  static LayoutDesc fromTensor(const ::tt::runtime::Tensor &tensor);

  LayoutDesc(const ::ttnn::StorageType &storageType,
             const ::ttnn::Layout &layout, const ::ttnn::DataType &dataType,
             const std::optional<::ttnn::MemoryConfig> &memoryConfig);

  bool isOnHost() const;
  bool isOnDevice() const;
  bool isTilized() const;

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

private:
  std::vector<std::uint32_t> programInputIds;
  std::vector<std::uint32_t> programOutputIds;
  TensorMap intermedTensors;
  TensorPtrMap liveTensors;

  const ::tt::runtime::Tensor &getRuntimeTensor(std::uint32_t globalId) const;
  ::tt::runtime::Tensor &getRuntimeTensor(std::uint32_t globalId);
};

class ProgramContext {
public:
  ProgramContext(const std::vector<uint32_t> &programInputIds,
                 const std::vector<uint32_t> &programOutputIds,
                 TensorPtrMap &&liveTensors,
                 common::DylibManager &&programDylibManager,
                 std::shared_ptr<::ttnn::MeshDevice> meshDevice,
                 const Binary &executableHandle, size_t programIndex = 0)
      : tensorPool(ProgramTensorPool(programInputIds, programOutputIds,
                                     std::move(liveTensors))),
        dylibManager(std::move(programDylibManager)), meshDevice(meshDevice),
        executableHandle(executableHandle), programIndex(programIndex) {
    LOG_ASSERT(meshDevice, "Submesh cannot be null");
  }

  ProgramContext(const ProgramContext &) = delete;
  ProgramContext &operator=(const ProgramContext &) = delete;
  ProgramContext(ProgramContext &&) = delete;
  ProgramContext &operator=(ProgramContext &&) = delete;

  //
  // Sub Mesh Operations
  //

  ::ttnn::MeshDevice &getMeshDevice() { return *meshDevice; }
  std::shared_ptr<::ttnn::MeshDevice> getMeshDevicePtr() { return meshDevice; }

  size_t meshDeviceSize() const { return meshDevice->num_devices(); }

  const ::ttnn::MeshShape &meshDeviceShape() const {
    return meshDevice->shape();
  }

  //
  // Dylib Manager Operation
  //
  void *tryGetDylibHandle(const uint32_t dylibId) {
    return dylibManager.getHandle(dylibId);
  }

  //
  // Tensor Pool Operations
  //
  ProgramTensorPool &getTensorPool() { return tensorPool; }

  const ProgramTensorPool &getTensorPool() const { return tensorPool; }

  //
  // Executable Handle Operations
  //
  std::shared_ptr<TensorCache> getCache() {
    return executableHandle.getCache();
  }

  Binary &getExecutableHandle() { return executableHandle; }

  //
  // Program Index getter
  //
  size_t getProgramIndex() const { return programIndex; }

private:
  ProgramTensorPool tensorPool;

  common::DylibManager dylibManager;

  std::shared_ptr<::ttnn::MeshDevice> meshDevice;

  // The executable binary handle
  Binary executableHandle;

  // The index of the program within the binary
  const size_t programIndex;
};

} // namespace tt::runtime::ttnn

#endif
