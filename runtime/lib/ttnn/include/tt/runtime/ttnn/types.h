// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_TYPES_H
#define TT_RUNTIME_TTNN_TYPES_H

#include "tt/runtime/detail/ttnn.h"
#include <optional>
#include <unordered_map>

namespace tt::runtime::ttnn {
using DeviceVariant = std::variant<std::reference_wrapper<::ttnn::Device>,
                                   std::reference_wrapper<::ttnn::MeshDevice>>;

struct LayoutDesc {
  ::ttnn::BufferType bufferType;
  ::ttnn::Layout layout;
  ::ttnn::DataType dataType;
  std::optional<::ttnn::MemoryConfig> memoryConfig;

  LayoutDesc(const ::ttnn::BufferType &bufferType, const ::ttnn::Layout &layout,
             const ::ttnn::DataType &dataType,
             const std::optional<::ttnn::MemoryConfig> &memoryConfig)
      : bufferType(bufferType), layout(layout), dataType(dataType),
        memoryConfig(memoryConfig) {}

  bool isOnHost() const {
    return bufferType == ::ttnn::BufferType::SYSTEM_MEMORY;
  }
  bool isOnDevice() const { return !isOnHost(); }
};

class ProgramTensorPool {
public:
  ProgramTensorPool(
      const std::unordered_map<uint32_t, ::ttnn::Tensor *> &liveTensors,
      const std::vector<uint32_t> &programInputs,
      const std::vector<uint32_t> &programOutputs)
      : programInputs(programInputs), programOutputs(programOutputs),
        liveTensors(liveTensors) {}
  ProgramTensorPool(const ProgramTensorPool &) = delete;
  ProgramTensorPool &operator=(const ProgramTensorPool &) = delete;
  ProgramTensorPool(ProgramTensorPool &&) = default;
  ProgramTensorPool &operator=(ProgramTensorPool &&) = default;

  std::pair<std::unordered_map<std::uint32_t, ::ttnn::Tensor *>::iterator, bool>
  try_emplace(std::uint32_t globalId, const ::ttnn::Tensor &tensor);

  std::pair<std::unordered_map<std::uint32_t, ::ttnn::Tensor *>::iterator, bool>
  insert_or_assign(std::uint32_t globalId, const ::ttnn::Tensor &tensor);

  ::ttnn::Tensor &at(std::uint32_t globalId);

  const ::ttnn::Tensor &at(std::uint32_t globalId) const;

  size_t erase(std::uint32_t globalId);

  std::vector<Tensor> gatherOutputTensors();

  bool contains(std::uint32_t globalId) const {
    return liveTensors.contains(globalId);
  }

  const std::vector<std::uint32_t> &getProgramInputs() const {
    return programInputs;
  }

  const std::vector<std::uint32_t> &getProgramOutputs() const {
    return programOutputs;
  }

private:
  std::vector<std::uint32_t> programInputs;
  std::vector<std::uint32_t> programOutputs;
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
      ::ttnn::MeshDevice *parentMesh);
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

  ::ttnn::Device &getDeviceFromSubMesh(uint32_t meshId, int physicalDeviceId);

  ::ttnn::Device &getDeviceIndexFromSubMesh(uint32_t meshId, int deviceIndex);

  DeviceVariant getTargetDevice(uint32_t meshId);

  //
  // Tensor Pool Operations
  //
  ProgramTensorPool &getTensorPool() { return tensorPool; }
  const ProgramTensorPool &getTensorPool() const { return tensorPool; }

private:
  ProgramTensorPool tensorPool;

  // Contains all devices borrowed from the user that are available to the
  // program
  ::ttnn::MeshDevice *parentMesh = nullptr;

  // Contains subMeshes of the parentMesh that are used by the program
  // Will be populated by GetDevice ops
  std::unordered_map<uint32_t, std::shared_ptr<::ttnn::MeshDevice>> subMeshes;
};
} // namespace tt::runtime::ttnn

#endif
