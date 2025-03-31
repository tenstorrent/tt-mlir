// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_TYPES_H
#define TT_RUNTIME_TTNN_TYPES_H

#include "tt/runtime/detail/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/types.h"

#include <optional>
#include <unordered_map>

namespace tt::runtime::ttnn {
using DeviceVariant = std::variant<std::reference_wrapper<::ttnn::IDevice>,
                                   std::reference_wrapper<::ttnn::MeshDevice>>;
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

class LayoutConverter {
public:
  LayoutDesc inputDesc;
  LayoutDesc outputDesc;
  bool shouldTilize = false;
  bool shouldUntilize = false;
  bool shouldTypecast = false;
  bool shouldToDevice = false;
  bool shouldToMemoryConfig = false;
  bool shouldFromDevice = false;

  LayoutConverter(const LayoutDesc &inputDesc, const LayoutDesc &outputDesc);
  ::ttnn::Tensor convertTensorLayout(const ::ttnn::Tensor &input,
                                     std::optional<DeviceVariant> targetDevice);

private:
  ::ttnn::Tensor toLayoutIfNeeded(const ::ttnn::Tensor &input);
  ::ttnn::Tensor typecastIfNeeded(const ::ttnn::Tensor &input);
  ::ttnn::Tensor toDeviceIfNeeded(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice,
                                  bool force = false);
  ::ttnn::Tensor toMemoryConfigIfNeeded(const ::ttnn::Tensor &input);
  ::ttnn::Tensor fromDeviceIfNeeded(const ::ttnn::Tensor &input);

  ::ttnn::Tensor
  handleHostInputNoLayoutNoTypecast(const ::ttnn::Tensor &input,
                                    std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  handleHostInputLayoutNoTypecast(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  handleHostInputNoLayoutTypecast(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  handleHostInputLayoutTypecast(const ::ttnn::Tensor &input,
                                std::optional<DeviceVariant> targetDevice);
  ::ttnn::Tensor
  convertHostTensorLayout(const ::ttnn::Tensor &input,
                          std::optional<DeviceVariant> targetDevice);

  ::ttnn::Tensor
  handleDeviceInputNoLayoutNoTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor handleDeviceInputLayoutNoTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor handleDeviceInputNoLayoutTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor handleDeviceInputLayoutTypecast(const ::ttnn::Tensor &input);
  ::ttnn::Tensor convertDeviceTensorLayout(const ::ttnn::Tensor &input);
};

class ProgramTensorPool {
public:
  using TensorMap = std::unordered_map<uint32_t, ::tt::runtime::Tensor>;
  using TensorMapIterator = typename TensorMap::iterator;
  ProgramTensorPool(
      const std::vector<uint32_t> &programInputIds,
      const std::vector<uint32_t> &programOutputIds,
      std::unordered_map<uint32_t, ::tt::runtime::Tensor> &&liveTensors)
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
  size_t getRuntimeTensorUseCount(std::uint32_t globalId) const;

  const ::ttnn::Tensor &getTTNNTensorAndValidate(
      const ::tt::target::ttnn::TensorRef *tensorRef) const;
  ::ttnn::Tensor &
  getTTNNTensorAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef);

  std::pair<TensorMapIterator, bool>
  insertAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef,
                    const ::ttnn::Tensor &ttnnTensor);

  std::vector<::tt::runtime::Tensor> gatherOutputTensors();

  TensorMapIterator erase(const ::tt::target::ttnn::TensorRef *tensorRef);

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
  TensorMap liveTensors;

  const ::tt::runtime::Tensor &getRuntimeTensor(std::uint32_t globalId) const;
};

class ProgramContext {
public:
  ProgramContext(
      const std::vector<uint32_t> &programInputIds,
      const std::vector<uint32_t> &programOutputIds,
      std::unordered_map<uint32_t, ::tt::runtime::Tensor> &&liveTensors,
      common::DylibManager &&programDylibManager,
      ::ttnn::MeshDevice *parentMesh)
      : tensorPool(ProgramTensorPool(programInputIds, programOutputIds,
                                     std::move(liveTensors))),
        dylibManager(std::move(programDylibManager)), parentMesh(parentMesh) {
    LOG_ASSERT(parentMesh, "Parent mesh cannot be null");
  }
  ProgramContext(const ProgramContext &) = delete;
  ProgramContext &operator=(const ProgramContext &) = delete;
  ProgramContext(ProgramContext &&) = default;
  ProgramContext &operator=(ProgramContext &&) = default;

  //
  // Parent Mesh Operations
  //
  ::ttnn::MeshDevice &getParentMesh() { return *parentMesh; }

  const ::ttnn::MeshDevice &getParentMesh() const { return *parentMesh; }

  size_t parentMeshSize() const { return parentMesh->num_devices(); }

  //
  // Sub Mesh Operations
  //
  void addSubMesh(uint32_t meshId, std::shared_ptr<::ttnn::MeshDevice> subMesh);

  ::ttnn::MeshDevice &getSubMesh(uint32_t meshId);

  size_t subMeshSize(uint32_t meshId) const;

  ::ttnn::IDevice &getDeviceFromSubMesh(uint32_t meshId, int physicalDeviceId);

  ::ttnn::IDevice &getDeviceIndexFromSubMesh(
      uint32_t meshId, ::tt::tt_metal::distributed::MeshCoordinate meshCoords);

  DeviceVariant getTargetDevice(uint32_t meshId);

  void *tryGetDylibHandle(const uint32_t dylibId) {
    return dylibManager.getHandle(dylibId);
  }

  //
  // Tensor Pool Operations
  //
  ProgramTensorPool &getTensorPool() { return tensorPool; }
  const ProgramTensorPool &getTensorPool() const { return tensorPool; }

private:
  ProgramTensorPool tensorPool;

  common::DylibManager dylibManager;
  // Contains all devices borrowed from the user that are available to the
  // program
  ::ttnn::MeshDevice *parentMesh = nullptr;

  // Contains subMeshes of the parentMesh that are used by the program
  // Will be populated by GetDevice ops
  std::unordered_map<uint32_t, std::shared_ptr<::ttnn::MeshDevice>> subMeshes;
};
} // namespace tt::runtime::ttnn

#endif
