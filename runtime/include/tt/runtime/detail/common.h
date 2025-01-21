// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_COMMON_H
#define TT_RUNTIME_DETAIL_COMMON_H

#include <optional>

#define FMT_HEADER_ONLY
#include "tt-metalium/tt_metal.hpp"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/types.h"


namespace tt::runtime::common {

inline ::tt::tt_metal::DispatchCoreType getDispatchCoreType(std::optional<DispatchCoreType> dispatchCoreType) {


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
    const BoardType boardType = ::tt::Cluster::instance().get_board_type(0);
    switch (boardType)
    {
    case BoardType::N150:
    case BoardType::GALAXY:
      type = ::tt::tt_metal::DispatchCoreType::WORKER;
      break;
    default:
      type = ::tt::tt_metal::DispatchCoreType::ETH;
      break;
    }
  }
  return type;
}

} // namespace tt::runtime::common
#endif // TT_RUNTIME_DETAIL_COMMON_H
