// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn {

using tt::runtime::DeviceRuntime;

//
// LayoutDesc APIs
//
LayoutDesc LayoutDesc::fromTensor(const ::tt::runtime::Tensor &tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  ::ttnn::StorageType storageType = ttnnTensor.storage_type();
  ::ttnn::Layout layout = ttnnTensor.get_layout();
  ::ttnn::DataType dtype = ttnnTensor.get_dtype();

  std::optional<::ttnn::MemoryConfig> memoryConfig = std::nullopt;
  if (storageType == ::ttnn::StorageType::DEVICE ||
      storageType == ::ttnn::StorageType::MULTI_DEVICE) {
    memoryConfig = ttnnTensor.memory_config();
  }

  return LayoutDesc(storageType, layout, dtype, memoryConfig);
}

LayoutDesc::LayoutDesc(const ::ttnn::StorageType &storageType,
                       const ::ttnn::Layout &layout,
                       const ::ttnn::DataType &dataType,
                       const std::optional<::ttnn::MemoryConfig> &memoryConfig)
    : storageType(storageType), layout(layout), dataType(dataType),
      memoryConfig(memoryConfig) {}

bool LayoutDesc::isOnHost() const {
  return (storageType == ::ttnn::StorageType::OWNED) ||
         (storageType == ::ttnn::StorageType::BORROWED) ||
         (storageType == ::ttnn::StorageType::MULTI_DEVICE_HOST);
}

bool LayoutDesc::isOnDevice() const { return !isOnHost(); }

bool LayoutDesc::isTilized() const { return layout == ::ttnn::Layout::TILE; }

bool LayoutDesc::operator==(const LayoutDesc &other) const {
  return (storageType == other.storageType) && (layout == other.layout) &&
         (dataType == other.dataType) && (memoryConfig == other.memoryConfig);
}

//
// ProgramTensorPool APIs
//

const ::tt::runtime::Tensor &
ProgramTensorPool::getRuntimeTensor(std::uint32_t globalId) const {
  auto it = liveTensors.find(globalId);
  LOG_ASSERT(it != liveTensors.end(), "Tensor not found in tensor pool");
  return *(it->second);
}

::tt::runtime::Tensor &
ProgramTensorPool::getRuntimeTensor(std::uint32_t globalId) {
  return const_cast<::tt::runtime::Tensor &>(
      static_cast<const ProgramTensorPool &>(*this).getRuntimeTensor(globalId));
}

const ::tt::runtime::Tensor &ProgramTensorPool::getRuntimeTensorAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef) const {
  LOG_ASSERT(tensorRef != nullptr, "tensorRef should not be null");
  const ::tt::runtime::Tensor &runtimeTensor =
      getRuntimeTensor(tensorRef->global_id());
  const ::ttnn::Tensor &ttnnTensor =
      runtimeTensor
          .as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  DEBUG_ASSERT(ttnnTensor.is_allocated());
  debug::checkTensorRefMatchesTTNNTensor(tensorRef, ttnnTensor);
  return runtimeTensor;
}

::tt::runtime::Tensor &ProgramTensorPool::getRuntimeTensorAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef) {
  return const_cast<::tt::runtime::Tensor &>(
      static_cast<const ProgramTensorPool &>(*this).getRuntimeTensorAndValidate(
          tensorRef));
}

const ::tt::runtime::ttnn::TTNNTensorWrapper &
ProgramTensorPool::getTTNNTensorWrapperAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef) const {
  const ::tt::runtime::Tensor &runtimeTensor =
      getRuntimeTensorAndValidate(tensorRef);
  return runtimeTensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
      DeviceRuntime::TTNN);
}

::tt::runtime::ttnn::TTNNTensorWrapper &
ProgramTensorPool::getTTNNTensorWrapperAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef) {
  return const_cast<::tt::runtime::ttnn::TTNNTensorWrapper &>(
      static_cast<const ProgramTensorPool &>(*this)
          .getTTNNTensorWrapperAndValidate(tensorRef));
}

const ::ttnn::Tensor &ProgramTensorPool::getTTNNTensorAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef) const {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      getTTNNTensorWrapperAndValidate(tensorRef);
  return tensorWrapper.getTensor();
}

::ttnn::Tensor &ProgramTensorPool::getTTNNTensorAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef) {
  return const_cast<::ttnn::Tensor &>(
      static_cast<const ProgramTensorPool &>(*this).getTTNNTensorAndValidate(
          tensorRef));
}

std::pair<TensorPtrMapIterator, bool>
ProgramTensorPool::insertTTNNTensorAndValidate(
    const ::tt::target::ttnn::TensorRef *tensorRef,
    const ::ttnn::Tensor &ttnnTensor, bool retain) {
  LOG_ASSERT(tensorRef != nullptr, "tensorRef should not be null");
  std::uint32_t globalId = tensorRef->global_id();
  DEBUG_ASSERT(ttnnTensor.is_allocated());
  debug::checkTensorRefMatchesTTNNTensor(tensorRef, ttnnTensor);

  ::tt::runtime::Tensor runtimeTensor =
      utils::createRuntimeTensorFromTTNN(ttnnTensor, retain);
  auto [iter, inserted] =
      intermedTensors.insert_or_assign(globalId, runtimeTensor);

  return liveTensors.insert_or_assign(globalId, &(iter->second));
}

std::vector<::tt::runtime::Tensor> ProgramTensorPool::gatherOutputTensors() {
  std::vector<::tt::runtime::Tensor> outputs;
  outputs.reserve(programOutputIds.size());
  std::transform(programOutputIds.begin(), programOutputIds.end(),
                 std::back_inserter(outputs), [this](std::uint32_t globalId) {
                   ::tt::runtime::Tensor &out = getRuntimeTensor(globalId);
                   ::tt::runtime::ttnn::TTNNTensorWrapper &ttnnTensor =
                       out.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
                           DeviceRuntime::TTNN);
                   ttnnTensor.setRetain(false);
                   return out;
                 });
  return outputs;
}

TensorPtrMapIterator
ProgramTensorPool::erase(const ::tt::target::ttnn::TensorRef *tensorRef) {
  LOG_ASSERT(tensorRef != nullptr, "tensorRef should not be null");
  std::uint32_t globalId = tensorRef->global_id();
  intermedTensors.erase(globalId);
  auto it = liveTensors.find(globalId);
  LOG_ASSERT(it != liveTensors.end(),
             "Tensor to erase not found in tensor pool");
  return liveTensors.erase(it);
}

//
// ProgramContext APIs
//
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

::ttnn::IDevice &ProgramContext::getDeviceFromSubMesh(uint32_t meshId,
                                                      int physicalDeviceId) {
  LOG_ASSERT(subMeshes.contains(meshId));
  auto &subMesh = *subMeshes.at(meshId);
  return *subMesh.get_device(physicalDeviceId);
}

::ttnn::IDevice &ProgramContext::getDeviceIndexFromSubMesh(
    uint32_t meshId, ::tt::tt_metal::distributed::MeshCoordinate meshCoords) {
  LOG_ASSERT(subMeshes.contains(meshId));
  auto &subMesh = *subMeshes.at(meshId);
  return *subMesh.get_device(meshCoords);
}

DeviceVariant ProgramContext::getTargetDevice(uint32_t meshId) {
  LOG_ASSERT(subMeshes.contains(meshId));
  auto &subMesh = *subMeshes.at(meshId);
  if (subMesh.num_devices() == 1) {
    return std::ref(
        *subMesh.get_device(::tt::tt_metal::distributed::MeshCoordinate(0, 0)));
  }
  return std::ref(subMesh);
}

} // namespace tt::runtime::ttnn
