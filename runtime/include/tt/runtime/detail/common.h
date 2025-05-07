// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_H
#define TT_RUNTIME_DETAIL_COMMON_H

#include <optional>

#define FMT_HEADER_ONLY
#include "tt-metalium/host_api.hpp"

#include "tt/runtime/detail/logger.h"
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

} // namespace tt::runtime::common
#endif // TT_RUNTIME_DETAIL_COMMON_H
