// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn {

//
// LayoutConverter APIs
//
LayoutConverter::LayoutConverter(const LayoutDesc &inputDesc,
                                 const LayoutDesc &outputDesc)
    : inputDesc(inputDesc), outputDesc(outputDesc) {
  shouldTilize = (inputDesc.layout == ::ttnn::Layout::ROW_MAJOR and
                  outputDesc.layout == ::ttnn::Layout::TILE);
  shouldUntilize = (inputDesc.layout == ::ttnn::Layout::TILE and
                    outputDesc.layout == ::ttnn::Layout::ROW_MAJOR);
  shouldTypecast = (inputDesc.dataType != outputDesc.dataType);
  shouldToDevice = (inputDesc.isOnHost() and outputDesc.isOnDevice());
  shouldToMemoryConfig = (not shouldToDevice and outputDesc.isOnDevice() and
                          (inputDesc.memoryConfig != outputDesc.memoryConfig));
  shouldFromDevice = (inputDesc.isOnDevice() and outputDesc.isOnHost());
}

::ttnn::Tensor LayoutConverter::convertTensorLayout(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  if (inputDesc.isOnHost()) {
    return convertHostTensorLayout(input, targetDevice);
  }
  return convertDeviceTensorLayout(input);
}

::ttnn::Tensor LayoutConverter::toLayoutIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldTilize) {
    return ::ttnn::to_layout(input, ::ttnn::Layout::TILE, std::nullopt,
                             std::nullopt,
                             static_cast<::ttnn::Device *>(nullptr));
  }
  if (shouldUntilize) {
    return ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                             std::nullopt,
                             static_cast<::ttnn::Device *>(nullptr));
  }
  return input;
}

::ttnn::Tensor LayoutConverter::typecastIfNeeded(const ::ttnn::Tensor &input) {
  if (not shouldTypecast) {
    return input;
  }
  if (utils::isOnHost(input.storage_type())) {
    return ::ttnn::to_dtype(input, outputDesc.dataType);
  }
  return ::ttnn::typecast(input, outputDesc.dataType);
}

::ttnn::Tensor
LayoutConverter::toDeviceIfNeeded(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice,
                                  bool force) {
  if (shouldToDevice or force) {
    LOG_ASSERT(targetDevice.has_value());
    return std::visit(
        [&](auto &&targetDevice) -> ::ttnn::Tensor {
          return ::ttnn::to_device(input, &(targetDevice.get()),
                                   outputDesc.memoryConfig);
        },
        targetDevice.value());
  }
  return input;
}

::ttnn::Tensor
LayoutConverter::toMemoryConfigIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldToMemoryConfig) {
    LOG_ASSERT(outputDesc.memoryConfig.has_value());
    return ::ttnn::to_memory_config(input, outputDesc.memoryConfig.value());
  }
  return input;
}

::ttnn::Tensor
LayoutConverter::fromDeviceIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldFromDevice) {
    return ::ttnn::from_device(input);
  }
  return input;
}

::ttnn::Tensor LayoutConverter::handleHostInputNoLayoutNoTypecast(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
  out = toMemoryConfigIfNeeded(out);
  return out;
}

::ttnn::Tensor LayoutConverter::handleHostInputLayoutNoTypecast(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  if (shouldUntilize) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize and outputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize and outputDesc.dataType != ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleHostInputNoLayoutTypecast(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  if (outputDesc.layout == ::ttnn::Layout::TILE) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (outputDesc.layout != ::ttnn::Layout::TILE) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleHostInputLayoutTypecast(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  if (shouldUntilize) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize and inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize and outputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize and inputDesc.dataType != ::ttnn::DataType::BFLOAT16 and
      outputDesc.dataType != ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::convertHostTensorLayout(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  bool shouldToLayout = (shouldTilize or shouldUntilize);
  LOG_ASSERT(not shouldToDevice or targetDevice.has_value(),
             "Target device must be provided for ToDevice");
  if (not shouldToLayout and not shouldTypecast) {
    return handleHostInputNoLayoutNoTypecast(input, targetDevice);
  }
  if (shouldToLayout and not shouldTypecast) {
    return handleHostInputLayoutNoTypecast(input, targetDevice);
  }
  if (not shouldToLayout and shouldTypecast) {
    return handleHostInputNoLayoutTypecast(input, targetDevice);
  }
  if (shouldToLayout and shouldTypecast) {
    return handleHostInputLayoutTypecast(input, targetDevice);
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleDeviceInputNoLayoutNoTypecast(
    const ::ttnn::Tensor &input) {
  ::ttnn::Tensor out = toMemoryConfigIfNeeded(input);
  out = fromDeviceIfNeeded(out);
  return out;
}

::ttnn::Tensor LayoutConverter::handleDeviceInputLayoutNoTypecast(
    const ::ttnn::Tensor &input) {
  if (shouldUntilize and shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize and not shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device untilize.");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  /* If we should tilize and the input data type is bfloat16, tilize on device
   */
  if (shouldTilize and inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  /* If we should tilize and the input data type is not bfloat16, tilize on
   * host */
  if (shouldTilize and inputDesc.dataType != ::ttnn::DataType::BFLOAT16 and
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldTilize and inputDesc.dataType != ::ttnn::DataType::BFLOAT16 and
      not shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device tilize.");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleDeviceInputNoLayoutTypecast(
    const ::ttnn::Tensor &input) {
  if (inputDesc.isTilized()) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(input);
    return out;
  }

  if (not inputDesc.isTilized() and shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = typecastIfNeeded(out);
    return out;
  }

  if (not inputDesc.isTilized() and not shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device typecast.");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor
LayoutConverter::handleDeviceInputLayoutTypecast(const ::ttnn::Tensor &input) {
  if (shouldUntilize and shouldFromDevice) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize and not shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device untilize.");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize and inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldTilize and inputDesc.dataType != ::ttnn::DataType::BFLOAT16 and
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    return out;
  }

  if (shouldTilize and inputDesc.dataType != ::ttnn::DataType::BFLOAT16 and
      not shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device tilize.");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor
LayoutConverter::convertDeviceTensorLayout(const ::ttnn::Tensor &input) {
  bool shouldToLayout = (shouldTilize or shouldUntilize);
  if (not shouldToLayout and not shouldTypecast) {
    return handleDeviceInputNoLayoutNoTypecast(input);
  }
  if (shouldToLayout and not shouldTypecast) {
    return handleDeviceInputLayoutNoTypecast(input);
  }
  if (not shouldToLayout and shouldTypecast) {
    return handleDeviceInputNoLayoutTypecast(input);
  }
  if (shouldToLayout and shouldTypecast) {
    return handleDeviceInputLayoutTypecast(input);
  }
  LOG_FATAL("Unreachable code path");
}

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
