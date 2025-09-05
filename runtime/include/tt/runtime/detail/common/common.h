// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_COMMON_H
#define TT_RUNTIME_DETAIL_COMMON_COMMON_H

#include <optional>

#define FMT_HEADER_ONLY
#include "tt-metalium/host_api.hpp"

#include "tt-metalium/fabric_types.hpp"
#include "tt/runtime/detail/common/flatbuffer_operator_ostream.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/types.h"

namespace tt::runtime::common {

inline ::tt::tt_metal::DispatchCoreType
getDispatchCoreType(std::optional<DispatchCoreType> dispatchCoreType) {

  ::tt::tt_metal::DispatchCoreType type;
  if (dispatchCoreType.has_value()) {
    if (dispatchCoreType == DispatchCoreType::ETH) {
      type = ::tt::tt_metal::DispatchCoreType::ETH;
    } else if (dispatchCoreType == DispatchCoreType::WORKER) {
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
toTTFabricConfig(tt::runtime::FabricConfig cfg) {
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
  default:
    LOG_FATAL("Unsupported data type");
  }
}

inline ::tt::runtime::Arch toRuntimeArch(::tt::ARCH arch) {
  switch (arch) {
  case ::tt::ARCH::GRAYSKULL:
    return ::tt::runtime::Arch::GRAYSKULL;
  case ::tt::ARCH::WORMHOLE_B0:
    return ::tt::runtime::Arch::WORMHOLE_B0;
  case ::tt::ARCH::BLACKHOLE:
    return ::tt::runtime::Arch::BLACKHOLE;
  case ::tt::ARCH::QUASAR:
    return ::tt::runtime::Arch::QUASAR;
  default:
    LOG_FATAL("Unsupported device architecture");
  }
}

inline UnpackToDestMode
toUnpackToDestMode(const tt::target::UnpackToDestMode &unpackToDestMode) {
  switch (unpackToDestMode) {
  case tt::target::UnpackToDestMode::Fp32:
    return UnpackToDestMode::UnpackToDestFp32;
  case tt::target::UnpackToDestMode::Default:
    return UnpackToDestMode::Default;
    LOG_FATAL("Unsupported unpack to dest mode");
  }
}

} // namespace tt::runtime::common
#endif // TT_RUNTIME_DETAIL_COMMON_COMMON_H
