// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_COMMON_H
#define TT_RUNTIME_DETAIL_COMMON_COMMON_H

#include <optional>

#include "ttmlir/Target/Common/Target.h"

#define FMT_HEADER_ONLY
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/mesh_device.hpp"

#include "tt-metalium/fabric_types.hpp"
#include "tt/runtime/detail/common/flatbuffer_operator_ostream.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/types.h"

namespace tt::runtime::common {

inline ::tt::tt_metal::DispatchCoreType getDispatchCoreType(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType) {

  ::tt::tt_metal::DispatchCoreType type;
  if (dispatchCoreType.has_value()) {
    if (dispatchCoreType == ::tt::runtime::DispatchCoreType::Ethernet) {
      type = ::tt::tt_metal::DispatchCoreType::ETH;
    } else if (dispatchCoreType == ::tt::runtime::DispatchCoreType::Worker) {
      type = ::tt::tt_metal::DispatchCoreType::WORKER;
    } else {
      LOG_FATAL("Unsupported dispatch core type");
    }
  } else {
    size_t numDevices = ::tt::tt_metal::GetNumAvailableDevices();
    size_t numPCIeDevices = ::tt::tt_metal::GetNumPCIeDevices();
    type = numDevices == numPCIeDevices
               ? ::tt::tt_metal::DispatchCoreType::WORKER
               : ::tt::tt_metal::DispatchCoreType::ETH;
  }
  return type;
}

inline ::tt::tt_fabric::FabricConfig
toMetalFabricConfig(tt::runtime::FabricConfig cfg) {
  switch (cfg) {
  case tt::runtime::FabricConfig::DISABLED:
    return ::tt::tt_fabric::FabricConfig::DISABLED;
  case tt::runtime::FabricConfig::FABRIC_1D:
    return ::tt::tt_fabric::FabricConfig::FABRIC_1D;
  case tt::runtime::FabricConfig::FABRIC_1D_RING:
    return ::tt::tt_fabric::FabricConfig::FABRIC_1D_RING;
  case tt::runtime::FabricConfig::FABRIC_2D:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D;
  case tt::runtime::FabricConfig::FABRIC_2D_TORUS_X:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X;
  case tt::runtime::FabricConfig::FABRIC_2D_TORUS_Y:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y;
  case tt::runtime::FabricConfig::FABRIC_2D_TORUS_XY:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
  case tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC;
  case tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_X:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_X;
  case tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_Y:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_Y;
  case tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY:
    return ::tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY;
  case tt::runtime::FabricConfig::CUSTOM:
    return ::tt::tt_fabric::FabricConfig::CUSTOM;
  }
  LOG_FATAL("Unknown tt::runtime::FabricConfig value");
}

inline CoreRangeSet toCoreRangeSet(
    const ::flatbuffers::Vector<const tt::target::Dim2dRange *> *coreRangeSet) {
  std::set<CoreRange> coreRanges;
  for (const ::tt::target::Dim2dRange *coreRange : *coreRangeSet) {
    CoreCoord start(coreRange->loc().x(), coreRange->loc().y());
    // End is inclusive
    CoreCoord end(coreRange->loc().x() + coreRange->size().x() - 1,
                  coreRange->loc().y() + coreRange->size().y() - 1);
    coreRanges.emplace(start, end);
  }
  return CoreRangeSet(coreRanges);
}

inline ::tt::DataFormat toDataFormat(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::tt::DataFormat::Float32;
  case ::tt::target::DataType::Float16:
    return ::tt::DataFormat::Float16;
  case ::tt::target::DataType::BFloat16:
    return ::tt::DataFormat::Float16_b;
  case ::tt::target::DataType::UInt32:
    return ::tt::DataFormat::UInt32;
  case ::tt::target::DataType::UInt16:
    return ::tt::DataFormat::UInt16;
  case ::tt::target::DataType::UInt8:
    return ::tt::DataFormat::UInt8;
  case ::tt::target::DataType::Int32:
    return ::tt::DataFormat::Int32;
  case ::tt::target::DataType::BFP_BFloat8:
    return ::tt::DataFormat::Bfp8_b;
  default:
    LOG_FATAL("Unsupported data type");
  }
}

inline ::tt::target::Arch toTargetArch(::tt::ARCH arch) {
  switch (arch) {
  case ::tt::ARCH::GRAYSKULL:
    return ::tt::target::Arch::Grayskull;
  case ::tt::ARCH::WORMHOLE_B0:
    return ::tt::target::Arch::Wormhole_b0;
  case ::tt::ARCH::BLACKHOLE:
    return ::tt::target::Arch::Blackhole;
  case ::tt::ARCH::QUASAR:
    LOG_FATAL("Quasar architecture is not supported");
  case ::tt::ARCH::Invalid:
    LOG_FATAL("Invalid architecture");
  }
}

inline UnpackToDestMode
toUnpackToDestMode(const tt::target::UnpackToDestMode &unpackToDestMode) {
  switch (unpackToDestMode) {
  case tt::target::UnpackToDestMode::Fp32:
    return UnpackToDestMode::UnpackToDestFp32;
  case tt::target::UnpackToDestMode::Default:
    return UnpackToDestMode::Default;
  }
}

inline std::vector<UnpackToDestMode>
toUnpackToDestModes(const ::flatbuffers::Vector<tt::target::UnpackToDestMode>
                        *unpackToDestModesFB) {
  // Metal asserts that unpack_to_dest_mode.size() == NUM_CIRCULAR_BUFFERS.
  std::vector<UnpackToDestMode> unpackToDestModes(NUM_CIRCULAR_BUFFERS,
                                                  UnpackToDestMode::Default);
  if (unpackToDestModesFB == nullptr) {
    return unpackToDestModes;
  }
  uint32_t modeIdx = 0;
  for (auto mode : *unpackToDestModesFB) {
    LOG_ASSERT(modeIdx < NUM_CIRCULAR_BUFFERS);
    unpackToDestModes[modeIdx] = toUnpackToDestMode(mode);
    ++modeIdx;
  }
  return unpackToDestModes;
}

inline std::shared_ptr<::tt::tt_metal::distributed::MeshDevice>
createFullMeshDevice(
    std::optional<::tt::runtime::DispatchCoreType> dispatchCoreType) {

  ::tt::tt_metal::DispatchCoreType type =
      tt::runtime::common::getDispatchCoreType(dispatchCoreType);

  return ::tt::tt_metal::distributed::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig(
          /*mesh_shape=*/std::nullopt),
      DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE,
      /*num_command_queues=*/1, type);
}

} // namespace tt::runtime::common
#endif // TT_RUNTIME_DETAIL_COMMON_COMMON_H
