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
  void syncVersion(const TTNNTensorWrapper &other) {
    version.store(other.getVersion(), std::memory_order_relaxed);
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
                 ::tt::runtime::Device deviceHandle,
                 const Binary &executableHandle, size_t programIndex = 0)
      : tensorPool(ProgramTensorPool(programInputIds, programOutputIds,
                                     std::move(liveTensors))),
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
  // Executable Handle Operations
  //
  std::shared_ptr<TensorCache> getConstEvalTensorCache() {
    return executableHandle.getConstEvalTensorCache();
  }

  Binary &getExecutableHandle() { return executableHandle; }

  //
  // Program Index getter
  //
  size_t getProgramIndex() const { return programIndex; }

private:
  ProgramTensorPool tensorPool;

  common::DylibManager dylibManager;

  ::tt::runtime::Device deviceHandle;

  // The executable binary handle
  Binary executableHandle;

  // The index of the program within the binary
  const size_t programIndex;
};

} // namespace tt::runtime::ttnn

#endif
