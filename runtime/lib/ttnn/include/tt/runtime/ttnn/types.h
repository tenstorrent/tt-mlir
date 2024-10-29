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

  void copyTensorToUserOutput(const ::ttnn::Tensor &srcTensor,
                              std::uint32_t outputGlobalId) {
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
                 ::ttnn::MeshDevice *meshDevice)
      : tensorPool(
            ProgramTensorPool(liveTensors, programInputs, programOutputs)),
        meshDevice(meshDevice) {}

  const ::ttnn::MeshDevice &getMeshDevice() const {
    assert(meshDevice && "Mesh device not initialized");
    return *meshDevice;
  }

  ::ttnn::MeshDeviceView &getMeshView(uint32_t globalId) {
    assert(meshViews.contains(globalId) &&
           "Mesh view with global id not initialized");
    return *(meshViews.at(globalId));
  }

  ProgramTensorPool &getTensorPool() { return tensorPool; }

  void addMeshView(uint32_t globalId,
                   std::unique_ptr<::ttnn::MeshDeviceView> view) {
    assert(not meshViews.contains(globalId) &&
           "Mesh view with globalId already set");
    meshViews.try_emplace(globalId, std::move(view));
  }

  ::ttnn::Device &getDeviceFromView(uint32_t globalId, int deviceId) {
    assert(meshViews.contains(globalId) && "Mesh view not initialized");
    ::tt::tt_metal::distributed::Coordinate deviceCoord =
        meshViews.at(globalId)->find_device(deviceId);
    return *(
        meshViews.at(globalId)->get_device(deviceCoord.row, deviceCoord.col));
  }

private:
  ProgramTensorPool tensorPool;

  // Contains all devices borrowed from the user that are available to the
  // program
  ::ttnn::MeshDevice *meshDevice = nullptr;

  // Contains various views of meshDevice that is used by the program
  // Will be populated by get_device ops
  std::unordered_map<uint32_t, std::unique_ptr<::ttnn::MeshDeviceView>>
      meshViews;
};
} // namespace tt::runtime::ttnn

#endif
