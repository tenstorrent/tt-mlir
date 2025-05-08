// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/ttnn/utils.h"

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
  if (storageType == ::ttnn::StorageType::DEVICE) {
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
  return (storageType == ::ttnn::StorageType::HOST) ||
         (storageType == ::ttnn::StorageType::MULTI_DEVICE_HOST);
}

bool LayoutDesc::isOnDevice() const { return !isOnHost(); }

bool LayoutDesc::isTilized() const { return layout == ::ttnn::Layout::TILE; }

bool LayoutDesc::operator==(const LayoutDesc &other) const {
  return (storageType == other.storageType) && (layout == other.layout) &&
         (dataType == other.dataType) && (memoryConfig == other.memoryConfig);
}

//
// LayoutConverter APIs
//
LayoutConverter::LayoutConverter(const LayoutDesc &inputDesc,
                                 const LayoutDesc &outputDesc)
    : inputDesc(inputDesc), outputDesc(outputDesc) {
  shouldTilize = (inputDesc.layout == ::ttnn::Layout::ROW_MAJOR &&
                  outputDesc.layout == ::ttnn::Layout::TILE);
  shouldUntilize = (inputDesc.layout == ::ttnn::Layout::TILE &&
                    outputDesc.layout == ::ttnn::Layout::ROW_MAJOR);
  shouldTypecast = (inputDesc.dataType != outputDesc.dataType);
  shouldToDevice = (inputDesc.isOnHost() && outputDesc.isOnDevice());
  shouldToMemoryConfig = (!shouldToDevice && outputDesc.isOnDevice() &&
                          (inputDesc.memoryConfig != outputDesc.memoryConfig));
  shouldFromDevice = (inputDesc.isOnDevice() && outputDesc.isOnHost());
}

::ttnn::Tensor
LayoutConverter::convertTensorLayout(const ::ttnn::Tensor &input,
                                     OptionalMeshDeviceRef targetDevice) {
  if (inputDesc.isOnHost()) {
    return convertHostTensorLayout(input, targetDevice);
  }
  return convertDeviceTensorLayout(input);
}

::ttnn::Tensor LayoutConverter::toLayoutIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldTilize) {
    return ::ttnn::to_layout(input, ::ttnn::Layout::TILE, std::nullopt,
                             std::nullopt,
                             static_cast<::ttnn::MeshDevice *>(nullptr));
  }
  if (shouldUntilize) {
    return ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                             std::nullopt,
                             static_cast<::ttnn::MeshDevice *>(nullptr));
  }
  return input;
}

::ttnn::Tensor LayoutConverter::typecastIfNeeded(const ::ttnn::Tensor &input) {
  if (!shouldTypecast) {
    return input;
  }
  if (utils::isOnHost(input.storage_type())) {
    return ::ttnn::to_dtype(input, outputDesc.dataType);
  }
  return ::ttnn::typecast(input, outputDesc.dataType);
}

::ttnn::Tensor
LayoutConverter::toDeviceIfNeeded(const ::ttnn::Tensor &input,
                                  OptionalMeshDeviceRef targetDevice,
                                  bool force) {
  if (shouldToDevice || force) {
    LOG_ASSERT(targetDevice.has_value());
    return ::ttnn::to_device(input, &(targetDevice.value().get()),
                             outputDesc.memoryConfig);
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
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
  ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
  out = toMemoryConfigIfNeeded(out);
  return out;
}

::ttnn::Tensor LayoutConverter::handleHostInputLayoutNoTypecast(
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
  if (shouldUntilize) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && outputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && outputDesc.dataType != ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleHostInputNoLayoutTypecast(
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
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
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
  if (shouldUntilize) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && outputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && inputDesc.dataType != ::ttnn::DataType::BFLOAT16 &&
      outputDesc.dataType != ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor
LayoutConverter::convertHostTensorLayout(const ::ttnn::Tensor &input,
                                         OptionalMeshDeviceRef targetDevice) {
  bool shouldToLayout = (shouldTilize || shouldUntilize);
  LOG_ASSERT(!shouldToDevice || targetDevice.has_value(),
             "Target device must be provided for ToDevice");
  if (!shouldToLayout && !shouldTypecast) {
    return handleHostInputNoLayoutNoTypecast(input, targetDevice);
  }
  if (shouldToLayout && !shouldTypecast) {
    return handleHostInputLayoutNoTypecast(input, targetDevice);
  }
  if (!shouldToLayout && shouldTypecast) {
    return handleHostInputNoLayoutTypecast(input, targetDevice);
  }
  if (shouldToLayout && shouldTypecast) {
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
  if (shouldUntilize && shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device untilize.");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  /* If we should tilize and the input data type is bfloat16, tilize on device
   */
  if (shouldTilize && inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  /* If we should tilize and the input data type is not bfloat16, tilize on
   * host */
  if (shouldTilize && inputDesc.dataType != ::ttnn::DataType::BFLOAT16 &&
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldTilize && inputDesc.dataType != ::ttnn::DataType::BFLOAT16 &&
      !shouldFromDevice) {
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
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (!inputDesc.isTilized() && shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = typecastIfNeeded(out);
    return out;
  }

  if (!inputDesc.isTilized() && !shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device typecast.");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor
LayoutConverter::handleDeviceInputLayoutTypecast(const ::ttnn::Tensor &input) {
  if (shouldUntilize && shouldFromDevice) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = fromDeviceIfNeeded(out);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !shouldFromDevice) {
    LOG_WARNING("Currently no constraint checking for on-device untilize.");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldTilize && inputDesc.dataType != ::ttnn::DataType::BFLOAT16 &&
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    return out;
  }

  if (shouldTilize && inputDesc.dataType != ::ttnn::DataType::BFLOAT16 &&
      !shouldFromDevice) {
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
  bool shouldToLayout = (shouldTilize || shouldUntilize);
  if (!shouldToLayout && !shouldTypecast) {
    return handleDeviceInputNoLayoutNoTypecast(input);
  }
  if (shouldToLayout && !shouldTypecast) {
    return handleDeviceInputLayoutNoTypecast(input);
  }
  if (!shouldToLayout && shouldTypecast) {
    return handleDeviceInputNoLayoutTypecast(input);
  }
  if (shouldToLayout && shouldTypecast) {
    return handleDeviceInputLayoutTypecast(input);
  }
  LOG_FATAL("Unreachable code path");
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

} // namespace tt::runtime::ttnn
