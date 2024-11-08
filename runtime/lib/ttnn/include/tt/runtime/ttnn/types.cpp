// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn {

//
// ProgramTensorPool APIs
//
std::pair<std::unordered_map<std::uint32_t, ::ttnn::Tensor *>::iterator, bool>
ProgramTensorPool::try_emplace(std::uint32_t globalId,
                               const ::ttnn::Tensor &tensor) {
  auto it = liveTensors.find(globalId);
  if (it != liveTensors.end()) {
    return std::make_pair(it, false);
  }
  LOG_ASSERT(!intermedTensors.contains(globalId));
  intermedTensors.try_emplace(globalId, tensor);
  return liveTensors.try_emplace(globalId, &intermedTensors.at(globalId));
}

std::pair<std::unordered_map<std::uint32_t, ::ttnn::Tensor *>::iterator, bool>
ProgramTensorPool::insert_or_assign(std::uint32_t globalId,
                                    const ::ttnn::Tensor &tensor) {
  intermedTensors.insert_or_assign(globalId, tensor);
  return liveTensors.insert_or_assign(globalId, &intermedTensors.at(globalId));
}

::ttnn::Tensor &ProgramTensorPool::at(std::uint32_t globalId) {
  LOG_ASSERT(liveTensors.contains(globalId));
  return *liveTensors.at(globalId);
}

const ::ttnn::Tensor &ProgramTensorPool::at(std::uint32_t globalId) const {
  LOG_ASSERT(liveTensors.contains(globalId));
  return *liveTensors.at(globalId);
}

size_t ProgramTensorPool::erase(std::uint32_t globalId) {
  LOG_ASSERT(liveTensors.contains(globalId) &&
             intermedTensors.contains(globalId));
  intermedTensors.erase(globalId);
  return liveTensors.erase(globalId);
}

std::vector<Tensor> ProgramTensorPool::gatherOutputTensors() {
  std::vector<Tensor> outputTensors;
  outputTensors.reserve(programOutputs.size());
  std::transform(
      programOutputs.begin(), programOutputs.end(),
      std::back_inserter(outputTensors), [this](uint32_t outputGlobalId) {
        return utils::createRuntimeTensorFromTTNN(this->at(outputGlobalId));
      });
  return outputTensors;
}

//
// ProgramContext APIs
//
ProgramContext::ProgramContext(
    const std::unordered_map<uint32_t, ::ttnn::Tensor *> &liveTensors,
    const std::vector<uint32_t> &programInputs,
    const std::vector<uint32_t> &programOutputs, ::ttnn::MeshDevice *parentMesh)
    : tensorPool(ProgramTensorPool(liveTensors, programInputs, programOutputs)),
      parentMesh(parentMesh) {
  LOG_ASSERT(parentMesh, "Parent mesh cannot be null");
}

void ProgramContext::addSubMesh(uint32_t meshId,
                                std::shared_ptr<::ttnn::MeshDevice> subMesh) {
  auto [it, inserted] = subMeshes.try_emplace(meshId, subMesh);
  LOG_ASSERT(inserted, "Submesh already exists");
}

::ttnn::MeshDevice &ProgramContext::getSubMesh(uint32_t meshId) {
  LOG_ASSERT(subMeshes.contains(meshId));
  return *subMeshes.at(meshId);
}

size_t ProgramContext::subMeshSize(uint32_t meshId) const {
  LOG_ASSERT(subMeshes.contains(meshId));
  return subMeshes.at(meshId)->num_devices();
}

::ttnn::Device &ProgramContext::getDeviceFromSubMesh(uint32_t meshId,
                                                     int physicalDeviceId) {
  LOG_ASSERT(subMeshes.contains(meshId));
  auto &subMesh = *subMeshes.at(meshId);
  return *subMesh.get_device(physicalDeviceId);
}

::ttnn::Device &ProgramContext::getDeviceIndexFromSubMesh(uint32_t meshId,
                                                          int deviceIndex) {
  LOG_ASSERT(subMeshes.contains(meshId));
  auto &subMesh = *subMeshes.at(meshId);
  return *subMesh.get_device_index(deviceIndex);
}

DeviceVariant ProgramContext::getTargetDevice(uint32_t meshId) {
  LOG_ASSERT(subMeshes.contains(meshId));
  auto &subMesh = *subMeshes.at(meshId);
  if (subMesh.num_devices() == 1) {
    return std::ref(*subMesh.get_device_index(0));
  }
  return std::ref(subMesh);
}

} // namespace tt::runtime::ttnn
