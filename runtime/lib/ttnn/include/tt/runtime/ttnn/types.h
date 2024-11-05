// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTNN_RUNTIME_TYPES_H
#define TTNN_RUNTIME_TYPES_H

#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn {

using TensorMap = std::unordered_map<uint32_t, ::ttnn::Tensor *>;
struct ProgramTensorPool {
  ProgramTensorPool(const TensorMap &liveTensors,
                    const std::unordered_set<uint32_t> &programInputs,
                    const std::unordered_set<uint32_t> &programOutputs)
      : programInputs(programInputs), programOutputs(programOutputs),
        liveTensors(liveTensors) {}
  ProgramTensorPool(const ProgramTensorPool &) = delete;
  ProgramTensorPool &operator=(const ProgramTensorPool &) = delete;
  ProgramTensorPool(ProgramTensorPool &&) = default;
  ProgramTensorPool &operator=(ProgramTensorPool &&) = default;

  auto try_emplace(std::uint32_t globalId, const ::ttnn::Tensor &tensor) {
    auto it = liveTensors.find(globalId);
    if (it != liveTensors.end()) {
      return std::make_pair(it, false);
    }
    assert(!intermedTensors.contains(globalId));
    intermedTensors.try_emplace(globalId, tensor);
    return liveTensors.try_emplace(globalId, &intermedTensors.at(globalId));
  }

  auto insert_or_assign(std::uint32_t globalId, const ::ttnn::Tensor &tensor) {
    intermedTensors.insert_or_assign(globalId, tensor);
    return liveTensors.insert_or_assign(globalId,
                                        &intermedTensors.at(globalId));
  }

  ::ttnn::Tensor &at(std::uint32_t globalId) {
    assert(liveTensors.contains(globalId));
    return *liveTensors.at(globalId);
  }

  size_t erase(std::uint32_t globalId) {
    assert(liveTensors.contains(globalId) &&
           intermedTensors.contains(globalId));
    intermedTensors.erase(globalId);
    return liveTensors.erase(globalId);
  }

  void copyTensorToUserOutput(std::uint32_t outputGlobalId,
                              const ::ttnn::Tensor &srcTensor) {
    assert(liveTensors.contains(outputGlobalId));
    assert(isUserOutput(outputGlobalId));
    ::ttnn::Tensor &outputTensor = *liveTensors.at(outputGlobalId);
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(srcTensor);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    size_t size = outputTensor.volume() * outputTensor.element_size();
    std::memcpy(dst, src, size);
  }

  bool contains(std::uint32_t globalId) const {
    return liveTensors.contains(globalId);
  }

  bool isUserOutput(std::uint32_t globalId) const {
    return programOutputs.contains(globalId);
  }

  const std::unordered_set<std::uint32_t> &getProgramInputs() const {
    return programInputs;
  }

  const std::unordered_set<std::uint32_t> &getProgramOutputs() const {
    return programOutputs;
  }

private:
  std::unordered_set<std::uint32_t> programInputs;
  std::unordered_set<std::uint32_t> programOutputs;
  // A superset of intermedTensors, containing pointers to all tensors created
  // by the program and the input/output tensors passed in by the user
  TensorMap liveTensors;

  // A subset of liveTensors, containing values of any intermediate tensors
  // created by the program
  std::unordered_map<std::uint32_t, ::ttnn::Tensor> intermedTensors;
};

class ProgramContext {
public:
  ProgramContext(const TensorMap &liveTensors,
                 const std::unordered_set<uint32_t> &programInputs,
                 const std::unordered_set<uint32_t> &programOutputs,
                 ::ttnn::MeshDevice *parentMesh)
      : tensorPool(
            ProgramTensorPool(liveTensors, programInputs, programOutputs)),
        parentMesh(parentMesh) {
    assert(parentMesh && "Mesh device cannot be null");
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
  void addSubMesh(uint32_t meshId,
                  std::shared_ptr<::ttnn::MeshDevice> subMesh) {
    auto [it, inserted] = subMeshes.try_emplace(meshId, subMesh);
    assert(inserted && "Submesh already exists");
  }

  ::ttnn::MeshDevice &getSubMesh(uint32_t meshId) {
    assert(subMeshes.contains(meshId));
    return *subMeshes.at(meshId);
  }

  size_t subMeshSize(uint32_t meshId) const {
    assert(subMeshes.contains(meshId));
    return subMeshes.at(meshId)->num_devices();
  }

  ::ttnn::Device &getDeviceFromSubMesh(uint32_t meshId, int physicalDeviceId) {
    assert(subMeshes.contains(meshId));
    auto &subMesh = *subMeshes.at(meshId);
    return *subMesh.get_device(physicalDeviceId);
  }

  ::ttnn::Device &getDeviceIndexFromSubMesh(uint32_t meshId, int deviceIndex) {
    assert(subMeshes.contains(meshId));
    auto &subMesh = *subMeshes.at(meshId);
    return *subMesh.get_device_index(deviceIndex);
  }

  std::variant<std::reference_wrapper<::ttnn::Device>,
               std::reference_wrapper<::ttnn::MeshDevice>>
  getTargetDevice(uint32_t meshId) {
    assert(subMeshes.contains(meshId));
    auto &subMesh = *subMeshes.at(meshId);
    if (subMesh.num_devices() == 1) {
      return std::ref(getDeviceIndexFromSubMesh(meshId, 0));
    }
    return std::ref(subMesh);
  }

  //
  // Tensor Pool Operations
  //
  ProgramTensorPool &getTensorPool() { return tensorPool; }

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
