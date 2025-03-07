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

  LayoutDesc(const ::ttnn::StorageType &storageType,
             const ::ttnn::Layout &layout, const ::ttnn::DataType &dataType,
             const std::optional<::ttnn::MemoryConfig> &memoryConfig)
      : storageType(storageType), layout(layout), dataType(dataType),
        memoryConfig(memoryConfig) {}

  bool isOnHost() const {
    return (storageType == ::ttnn::StorageType::OWNED) ||
           (storageType == ::ttnn::StorageType::BORROWED) ||
           (storageType == ::ttnn::StorageType::MULTI_DEVICE_HOST);
  }
  bool isOnDevice() const { return !isOnHost(); }

  bool isTilized() const { return layout == ::ttnn::Layout::TILE; }
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
  ProgramTensorPool(
      const std::unordered_map<uint32_t, ::ttnn::Tensor *> &liveTensors,
      const std::vector<uint32_t> &programInputIds,
      const std::vector<uint32_t> &programOutputIds)
      : programInputIds(programInputIds), programOutputIds(programOutputIds),
        liveTensors(liveTensors) {}
  ProgramTensorPool(const ProgramTensorPool &) = delete;
  ProgramTensorPool &operator=(const ProgramTensorPool &) = delete;
  ProgramTensorPool(ProgramTensorPool &&) = default;
  ProgramTensorPool &operator=(ProgramTensorPool &&) = default;

  const ::ttnn::Tensor &
  getAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef) const;
  ::ttnn::Tensor &
  getAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef);

  std::pair<std::unordered_map<std::uint32_t, ::ttnn::Tensor *>::iterator, bool>
  insertAndValidate(const ::tt::target::ttnn::TensorRef *tensorRef,
                    const ::ttnn::Tensor &ttnnTensor);

  size_t erase(const ::tt::target::ttnn::TensorRef *tensorRef);

  std::vector<Tensor> gatherOutputTensors();

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
  // A superset of intermedTensors, containing pointers to all tensors created
  // by the program and the input tensors passed in by the user
  std::unordered_map<uint32_t, ::ttnn::Tensor *> liveTensors;

  // A subset of liveTensors, containing values of any intermediate tensors
  // created by the program
  std::unordered_map<std::uint32_t, ::ttnn::Tensor> intermedTensors;
};

class ProgramContext {
public:
  ProgramContext(
      const std::unordered_map<uint32_t, ::ttnn::Tensor *> &liveTensors,
      const std::vector<uint32_t> &programInputs,
      const std::vector<uint32_t> &programOutputs,
      common::DylibManager&& programDylibManager, ::ttnn::MeshDevice *parentMesh)
      : tensorPool(
            ProgramTensorPool(liveTensors, programInputs, programOutputs)),
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
