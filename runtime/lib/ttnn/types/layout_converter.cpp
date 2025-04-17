// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/ttnn/layout_converter.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn {

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

::ttnn::Tensor LayoutConverter::convertTensorLayout(
    const ::ttnn::Tensor &input, std::optional<DeviceVariant> targetDevice) {
  if (inputDesc.isOnHost()) {
    return convertHostTensorLayout(input, targetDevice);
  }
  return convertDeviceTensorLayout(input);
}

bool LayoutConverter::canTilizeDataTypeOnDevice(
    const ::ttnn::DataType &dataType) const {
  return dataType == ::ttnn::DataType::BFLOAT16 ||
         dataType == ::ttnn::DataType::FLOAT32;
}

bool LayoutConverter::canUntilizeDataTypeOnDevice(
    const ::ttnn::DataType &dataType) const {
  return dataType == ::ttnn::DataType::BFLOAT16 ||
         dataType == ::ttnn::DataType::FLOAT32;
}

::ttnn::Tensor LayoutConverter::toLayoutIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldTilize) {
    return ::ttnn::to_layout(input, ::ttnn::Layout::TILE, std::nullopt,
                             std::nullopt,
                             static_cast<::ttnn::IDevice *>(nullptr));
  }
  if (shouldUntilize) {
    return ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                             std::nullopt,
                             static_cast<::ttnn::IDevice *>(nullptr));
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
                                  std::optional<DeviceVariant> targetDevice,
                                  bool force) {
  if (shouldToDevice || force) {
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
  if (shouldUntilize && canUntilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !canUntilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && canTilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && !canTilizeDataTypeOnDevice(outputDesc.dataType)) {
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
  if (shouldUntilize && canUntilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = typecastIfNeeded(out);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !canUntilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && canTilizeDataTypeOnDevice(inputDesc.dataType)) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && canTilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && !canTilizeDataTypeOnDevice(inputDesc.dataType) &&
      !canTilizeDataTypeOnDevice(outputDesc.dataType)) {
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
  if (shouldUntilize && canUntilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !canUntilizeDataTypeOnDevice(outputDesc.dataType) &&
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !canUntilizeDataTypeOnDevice(outputDesc.dataType) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device untilize "
              "for output data type: ",
              debug::toString(outputDesc.dataType));
  }

  /* If we should tilize and the input data type is device tilizable, tilize on
   * device
   */
  if (shouldTilize && canTilizeDataTypeOnDevice(inputDesc.dataType)) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  /* If we should tilize and the input data type is not device tilizable, tilize
   * on host */
  if (shouldTilize && !canTilizeDataTypeOnDevice(inputDesc.dataType) &&
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldTilize && !canTilizeDataTypeOnDevice(inputDesc.dataType) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device tilize "
              "for input data type: ",
              debug::toString(inputDesc.dataType));
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
    LOG_FATAL("Currently to_layout does not support device to device typecast "
              "for input layout: ",
              debug::toString(inputDesc.layout));
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor
LayoutConverter::handleDeviceInputLayoutTypecast(const ::ttnn::Tensor &input) {
  if (shouldUntilize && canUntilizeDataTypeOnDevice(outputDesc.dataType)) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !canUntilizeDataTypeOnDevice(outputDesc.dataType) &&
      shouldFromDevice) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = fromDeviceIfNeeded(out);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !canUntilizeDataTypeOnDevice(outputDesc.dataType) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device untilize "
              "and typecast for output data type: ",
              debug::toString(outputDesc.dataType));
  }

  if (shouldTilize && canTilizeDataTypeOnDevice(inputDesc.dataType)) {
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldTilize && !canTilizeDataTypeOnDevice(inputDesc.dataType) &&
      shouldFromDevice) {
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    return out;
  }

  if (shouldTilize && !canTilizeDataTypeOnDevice(inputDesc.dataType) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device tilize "
              "and typecast for input data type: ",
              debug::toString(inputDesc.dataType));
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

} // namespace tt::runtime::ttnn
