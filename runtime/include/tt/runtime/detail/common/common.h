// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_COMMON_H
#define TT_RUNTIME_DETAIL_COMMON_COMMON_H

#include <optional>
#include <tt-metalium/fabric_types.hpp>

#define FMT_HEADER_ONLY
#include "tt-metalium/host_api.hpp"

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

[[nodiscard]] inline std::string trim_and_lower(std::string s) {
  auto first = std::find_if_not(
      s.begin(), s.end(), [](unsigned char ch) { return std::isspace(ch); });
  if (first == s.end()) {
    return {};
  }
  auto last = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch) {
                return std::isspace(ch);
              }).base();
  s.erase(last, s.end());
  s.erase(s.begin(), first);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return s;
}

inline ::tt::tt_metal::FabricConfig
getFabricConfig(const std::string &fabricConfig) {
  std::string s = trim_and_lower(fabricConfig);
  if (s == "disabled") {
    return ::tt::tt_metal::FabricConfig::DISABLED;
  }
  if (s == "fabric_1d") {
    return ::tt::tt_metal::FabricConfig::FABRIC_1D;
  }
  if (s == "fabric_1d_ring") {
    return ::tt::tt_metal::FabricConfig::FABRIC_1D_RING;
  }
  if (s == "fabric_2d") {
    return ::tt::tt_metal::FabricConfig::FABRIC_2D;
  }
  if (s == "fabric_2d_torus") {
    return ::tt::tt_metal::FabricConfig::FABRIC_2D_TORUS;
  }
  if (s == "fabric_2d_dynamic") {
    return ::tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC;
  }
  if (s == "custom") {
    return ::tt::tt_metal::FabricConfig::CUSTOM;
  }
  return ::tt::tt_metal::FabricConfig::DISABLED;
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

} // namespace tt::runtime::common
#endif // TT_RUNTIME_DETAIL_COMMON_COMMON_H
