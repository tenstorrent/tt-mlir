// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/test/utils.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::test {
Layout getDramInterleavedTileLayout(::tt::target::DataType dataType) {
  ::ttnn::DataType ttnnDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(dataType);
  ::tt::runtime::ttnn::LayoutDesc layoutDesc(::ttnn::BufferType::DRAM,
                                             ::ttnn::Layout::TILE, ttnnDataType,
                                             std::nullopt);
  return Layout(
      std::static_pointer_cast<void>(
          std::make_shared<::tt::runtime::ttnn::LayoutDesc>(layoutDesc)),
      ::tt::runtime::DeviceRuntime::TTNN);
}
Layout getDramInterleavedRowMajorLayout(::tt::target::DataType dataType) {
  ::ttnn::DataType ttnnDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(dataType);
  ::tt::runtime::ttnn::LayoutDesc layoutDesc(::ttnn::BufferType::DRAM,
                                             ::ttnn::Layout::ROW_MAJOR,
                                             ttnnDataType, std::nullopt);
  return Layout(
      std::static_pointer_cast<void>(
          std::make_shared<::tt::runtime::ttnn::LayoutDesc>(layoutDesc)),
      ::tt::runtime::DeviceRuntime::TTNN);
}
::tt::runtime::Layout getHostRowMajorLayout(::tt::target::DataType dataType) {
  ::ttnn::DataType ttnnDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(dataType);
  ::tt::runtime::ttnn::LayoutDesc layoutDesc(::ttnn::BufferType::SYSTEM_MEMORY,
                                             ::ttnn::Layout::ROW_MAJOR,
                                             ttnnDataType, std::nullopt);
  return Layout(
      std::static_pointer_cast<void>(
          std::make_shared<::tt::runtime::ttnn::LayoutDesc>(layoutDesc)),
      ::tt::runtime::DeviceRuntime::TTNN);
}
} // namespace tt::runtime::ttnn::test
