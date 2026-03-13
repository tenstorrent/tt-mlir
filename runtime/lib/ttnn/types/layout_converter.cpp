// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/layout_converter.h"
#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <fstream>

namespace tt::runtime::ttnn {

static size_t getProcessRSSBytes() {
  std::ifstream statm("/proc/self/statm");
  if (!statm.is_open()) {
    return 0;
  }
  size_t totalPages = 0;
  size_t residentPages = 0;
  statm >> totalPages >> residentPages;
  long pageSize = sysconf(_SC_PAGESIZE);
  return residentPages * static_cast<size_t>(pageSize);
}

static std::string tensorInfoStr(const ::ttnn::Tensor &t) {
  auto shape = t.logical_shape();
  std::string shapeStr = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      shapeStr += "x";
    }
    shapeStr += std::to_string(shape[i]);
  }
  shapeStr += "]";
  return "shape=" + shapeStr +
         " volume=" + std::to_string(t.physical_volume()) +
         " elem_size=" + std::to_string(t.element_size()) +
         " storage=" + debug::toString(t.storage_type()) +
         " layout=" + debug::toString(t.layout()) +
         " dtype=" + debug::toString(t.dtype());
}

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
  size_t rssBefore = getProcessRSSBytes();
  LOG_INFO("[LayoutConvMem] convertTensorLayout START ", tensorInfoStr(input),
           " shouldTilize=", shouldTilize, " shouldUntilize=", shouldUntilize,
           " shouldTypecast=", shouldTypecast,
           " shouldToDevice=", shouldToDevice,
           " shouldToMemoryConfig=", shouldToMemoryConfig,
           " shouldFromDevice=", shouldFromDevice,
           " rss_before=", rssBefore / 1024, "KB");

  ::ttnn::Tensor result;
  if (inputDesc.isOnHost()) {
    result = convertHostTensorLayout(input, targetDevice);
  } else {
    result = convertDeviceTensorLayout(input);
  }

  size_t rssAfter = getProcessRSSBytes();
  LOG_INFO("[LayoutConvMem] convertTensorLayout END ", tensorInfoStr(result),
           " rss_after=", rssAfter / 1024, "KB",
           " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
           "KB");
  return result;
}

::ttnn::Tensor LayoutConverter::toLayoutIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldTilize) {
    size_t rssBefore = getProcessRSSBytes();
    auto out = ::ttnn::to_layout(input, ::ttnn::Layout::TILE, std::nullopt,
                                 std::nullopt);
    size_t rssAfter = getProcessRSSBytes();
    LOG_INFO("[LayoutConvMem]   toLayout(TILIZE) ", tensorInfoStr(input),
             " -> ", tensorInfoStr(out),
             " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
             "KB");
    return out;
  }
  if (shouldUntilize) {
    size_t rssBefore = getProcessRSSBytes();
    auto out = ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                                 std::nullopt);
    size_t rssAfter = getProcessRSSBytes();
    LOG_INFO("[LayoutConvMem]   toLayout(UNTILIZE) ", tensorInfoStr(input),
             " -> ", tensorInfoStr(out),
             " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
             "KB");
    return out;
  }
  return input;
}

::ttnn::Tensor LayoutConverter::typecastIfNeeded(const ::ttnn::Tensor &input) {
  if (!shouldTypecast) {
    return input;
  }
  size_t rssBefore = getProcessRSSBytes();
  auto out = ::ttnn::typecast(input, outputDesc.dataType);
  size_t rssAfter = getProcessRSSBytes();
  LOG_INFO("[LayoutConvMem]   typecast ", tensorInfoStr(input), " -> ",
           tensorInfoStr(out),
           " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
           "KB");
  return out;
}

::ttnn::Tensor
LayoutConverter::toDeviceIfNeeded(const ::ttnn::Tensor &input,
                                  OptionalMeshDeviceRef targetDevice,
                                  bool force) {
  if (shouldToDevice || force) {
    LOG_ASSERT(targetDevice.has_value());
    size_t rssBefore = getProcessRSSBytes();
    auto out = ::ttnn::to_device(input, &(targetDevice.value().get()),
                                 outputDesc.memoryConfig);
    size_t rssAfter = getProcessRSSBytes();
    LOG_INFO("[LayoutConvMem]   toDevice ", tensorInfoStr(input), " -> ",
             tensorInfoStr(out),
             " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
             "KB");
    return out;
  }
  return input;
}

::ttnn::Tensor
LayoutConverter::toMemoryConfigIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldToMemoryConfig) {
    LOG_ASSERT(outputDesc.memoryConfig.has_value());
    size_t rssBefore = getProcessRSSBytes();
    auto out = ::ttnn::to_memory_config(input, outputDesc.memoryConfig.value());
    size_t rssAfter = getProcessRSSBytes();
    LOG_INFO("[LayoutConvMem]   toMemoryConfig ", tensorInfoStr(input), " -> ",
             tensorInfoStr(out),
             " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
             "KB");
    return out;
  }
  return input;
}

::ttnn::Tensor
LayoutConverter::fromDeviceIfNeeded(const ::ttnn::Tensor &input) {
  if (shouldFromDevice) {
    size_t rssBefore = getProcessRSSBytes();
    auto out = ::ttnn::from_device(input);
    size_t rssAfter = getProcessRSSBytes();
    LOG_INFO("[LayoutConvMem]   fromDevice ", tensorInfoStr(input), " -> ",
             tensorInfoStr(out),
             " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
             "KB");
    return out;
  }
  return input;
}

::ttnn::Tensor LayoutConverter::handleHostInputNoLayoutNoTypecast(
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
  LOG_INFO("[LayoutConvMem]  path=HostInput_NoLayout_NoTypecast");
  ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
  out = toMemoryConfigIfNeeded(out);
  return out;
}

::ttnn::Tensor LayoutConverter::handleHostInputLayoutNoTypecast(
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
  if (shouldUntilize && utils::canUntilizeOnDevice(outputDesc.dataType,
                                                   outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Untilize_OnDevice");
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !utils::canUntilizeOnDevice(outputDesc.dataType,
                                                    outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Untilize_OnHost");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      utils::canTilizeOnDevice(outputDesc.dataType, outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Tilize_OnDevice");
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize && (!utils::canTilizeOnDevice(outputDesc.dataType,
                                                 outputDesc.memoryConfig))) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Tilize_OnHost");
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
    LOG_INFO("[LayoutConvMem]  path=HostInput_NoLayout_Typecast_TileOnDevice");
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (outputDesc.layout != ::ttnn::Layout::TILE) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_NoLayout_Typecast_RMOnHost");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }
  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleHostInputLayoutTypecast(
    const ::ttnn::Tensor &input, OptionalMeshDeviceRef targetDevice) {
  if (shouldUntilize && utils::canUntilizeOnDevice(outputDesc.dataType,
                                                   outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Untilize_Typecast_OnDevice");
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = typecastIfNeeded(out);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldUntilize && !utils::canUntilizeOnDevice(outputDesc.dataType,
                                                    outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Untilize_Typecast_OnHost");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      utils::canTilizeOnDevice(inputDesc.dataType, outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Tilize_Typecast_InputDtOnDevice");
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      utils::canTilizeOnDevice(outputDesc.dataType, outputDesc.memoryConfig)) {
    LOG_INFO(
        "[LayoutConvMem]  path=HostInput_Tilize_Typecast_OutputDtOnDevice");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toDeviceIfNeeded(out, targetDevice);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      ((!utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig) &&
        !utils::canTilizeOnDevice(outputDesc.dataType,
                                  outputDesc.memoryConfig)))) {
    LOG_INFO("[LayoutConvMem]  path=HostInput_Tilize_Typecast_AllOnHost");
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
  LOG_INFO("[LayoutConvMem]  path=DeviceInput_NoLayout_NoTypecast");
  ::ttnn::Tensor out = toMemoryConfigIfNeeded(input);
  out = fromDeviceIfNeeded(out);
  return out;
}

::ttnn::Tensor LayoutConverter::handleDeviceInputLayoutNoTypecast(
    const ::ttnn::Tensor &input) {
  if (shouldUntilize && utils::canUntilizeOnDevice(outputDesc.dataType,
                                                   outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Untilize_OnDevice");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldUntilize &&
      !utils::canUntilizeOnDevice(outputDesc.dataType,
                                  outputDesc.memoryConfig) &&
      shouldFromDevice) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Untilize_OnHost");
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize &&
      !utils::canUntilizeOnDevice(outputDesc.dataType,
                                  outputDesc.memoryConfig) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device untilize "
              "for output data type or memory layout: ",
              debug::toString(outputDesc.dataType));
  }

  if (shouldTilize &&
      utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Tilize_OnDevice");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      (!utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig)) &&
      shouldFromDevice) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Tilize_OnHost");
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      (!utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig)) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device tilize "
              "for input data type or memory layout: ",
              debug::toString(inputDesc.dataType));
  }

  LOG_FATAL("Unreachable code path");
}

::ttnn::Tensor LayoutConverter::handleDeviceInputNoLayoutTypecast(
    const ::ttnn::Tensor &input) {
  if (inputDesc.isTilized()) {
    LOG_INFO(
        "[LayoutConvMem]  path=DeviceInput_NoLayout_Typecast_TiledOnDevice");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (!inputDesc.isTilized() && shouldFromDevice) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_NoLayout_Typecast_RMOnHost");
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
  if (shouldUntilize && utils::canUntilizeOnDevice(outputDesc.dataType,
                                                   outputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Untilize_Typecast_OnDevice");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldUntilize &&
      !utils::canUntilizeOnDevice(outputDesc.dataType,
                                  outputDesc.memoryConfig) &&
      shouldFromDevice) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Untilize_Typecast_OnHost");
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = fromDeviceIfNeeded(out);
    out = toLayoutIfNeeded(out);
    return out;
  }

  if (shouldUntilize &&
      !utils::canUntilizeOnDevice(outputDesc.dataType,
                                  outputDesc.memoryConfig) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device untilize "
              "and typecast for output data type or memory layout: ",
              debug::toString(outputDesc.dataType));
  }

  if (shouldTilize &&
      utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig)) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Tilize_Typecast_OnDevice");
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = typecastIfNeeded(out);
    out = toMemoryConfigIfNeeded(out);
    out = fromDeviceIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      (!utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig)) &&
      shouldFromDevice) {
    LOG_INFO("[LayoutConvMem]  path=DeviceInput_Tilize_Typecast_OnHost");
    ::ttnn::Tensor out = fromDeviceIfNeeded(input);
    out = toLayoutIfNeeded(out);
    out = typecastIfNeeded(out);
    return out;
  }

  if (shouldTilize &&
      (!utils::canTilizeOnDevice(inputDesc.dataType, inputDesc.memoryConfig)) &&
      !shouldFromDevice) {
    LOG_FATAL("Currently to_layout does not support device to device tilize "
              "and typecast for input data type or memory layout: ",
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
