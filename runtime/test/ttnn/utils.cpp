// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/test/ttnn/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/trace_cache.h"
#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

namespace tt::runtime::test::ttnn {
using ::tt::runtime::DeviceRuntime;
Layout getDramInterleavedTileLayout(::tt::target::DataType dataType) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);
  ::ttnn::DataType ttnnDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(dataType);
  ::tt::runtime::ttnn::LayoutDesc layoutDesc(::ttnn::StorageType::DEVICE,
                                             ::ttnn::Layout::TILE, ttnnDataType,
                                             ::ttnn::DRAM_MEMORY_CONFIG);
  return Layout(
      std::static_pointer_cast<void>(
          std::make_shared<::tt::runtime::ttnn::LayoutDesc>(layoutDesc)),
      DeviceRuntime::TTNN);
}
Layout getDramInterleavedRowMajorLayout(::tt::target::DataType dataType) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);
  ::ttnn::DataType ttnnDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(dataType);
  ::tt::runtime::ttnn::LayoutDesc layoutDesc(
      ::ttnn::StorageType::DEVICE, ::ttnn::Layout::ROW_MAJOR, ttnnDataType,
      ::ttnn::DRAM_MEMORY_CONFIG);
  return Layout(
      std::static_pointer_cast<void>(
          std::make_shared<::tt::runtime::ttnn::LayoutDesc>(layoutDesc)),
      DeviceRuntime::TTNN);
}
::tt::runtime::Layout getHostRowMajorLayout(::tt::target::DataType dataType) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);
  ::ttnn::DataType ttnnDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(dataType);
  ::tt::runtime::ttnn::LayoutDesc layoutDesc(::ttnn::StorageType::HOST,
                                             ::ttnn::Layout::ROW_MAJOR,
                                             ttnnDataType, std::nullopt);
  return Layout(
      std::static_pointer_cast<void>(
          std::make_shared<::tt::runtime::ttnn::LayoutDesc>(layoutDesc)),
      DeviceRuntime::TTNN);
}

std::optional<size_t> getTraceCacheDebugStat(::tt::runtime::Device device,
                                             const std::string &statName) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);
  auto traceCache =
      device.getTraceCache()->asSharedPtr<tt::runtime::ttnn::TraceCache>(
          DeviceRuntime::TTNN);
  return traceCache->getDebugStat(statName);
}
} // namespace tt::runtime::test::ttnn
