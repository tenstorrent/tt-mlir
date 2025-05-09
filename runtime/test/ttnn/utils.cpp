// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/test/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::test {
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

bool isProgramCacheEnabled(::tt::runtime::Device device) {
  LOG_ASSERT(getCurrentRuntime() == DeviceRuntime::TTNN);
  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return meshDevice.get_program_cache().is_enabled();
}
} // namespace tt::runtime::ttnn::test
