// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"

#include <set>

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/system_mesh.h"

namespace tt::runtime::common {

MeshFabricConfig
computeFabricConfig(const std::vector<::tt::target::ChipChannel> &chipChannels,
                    const std::vector<uint32_t> &meshShape,
                    const std::vector<int> &deviceIds) {
  LOG_ASSERT(meshShape.size() == 2,
             "meshShape must have exactly 2 dimensions, got ",
             meshShape.size());

  uint32_t numRows = meshShape[0];
  uint32_t numCols = meshShape[1];
  uint32_t totalDevices = numRows * numCols;

  if (totalDevices <= 1) {
    return {FabricConfig::DISABLED, {}};
  }

  LOG_ASSERT(deviceIds.size() == totalDevices, "Expected ", totalDevices,
             " device IDs, got ", deviceIds.size());

  std::set<std::pair<uint32_t, uint32_t>> connections;
  for (const auto &channel : chipChannels) {
    uint32_t id0 = channel.device_id0();
    uint32_t id1 = channel.device_id1();
    if (id0 > id1) {
      std::swap(id0, id1);
    }
    connections.insert({id0, id1});
  }

  auto getDeviceAt = [&deviceIds, numCols](uint32_t row,
                                           uint32_t col) -> uint32_t {
    int id = deviceIds[row * numCols + col];
    return id >= 0 ? static_cast<uint32_t>(id) : 0;
  };

  auto areConnected = [&connections](uint32_t id0, uint32_t id1) {
    if (id0 > id1) {
      std::swap(id0, id1);
    }
    return connections.count({id0, id1}) > 0;
  };

  // Check row wraparound: for each row, check if first and last device
  // connect. Default to linear (false) — only upgrade to ring if all rows have
  // wraparound.
  bool allRowsRing = false;
  if (numCols > 1) {
    allRowsRing = true;
    for (uint32_t row = 0; row < numRows && allRowsRing; ++row) {
      uint32_t firstDevice = getDeviceAt(row, 0);
      uint32_t lastDevice = getDeviceAt(row, numCols - 1);
      if (!areConnected(firstDevice, lastDevice)) {
        allRowsRing = false;
      }
    }
  }

  // Check column wraparound: for each column, check if first and last connect.
  // Default to linear (false) — only upgrade to ring if all columns have
  // wraparound.
  bool allColsRing = false;
  if (numRows > 1) {
    allColsRing = true;
    for (uint32_t col = 0; col < numCols && allColsRing; ++col) {
      uint32_t firstDevice = getDeviceAt(0, col);
      uint32_t lastDevice = getDeviceAt(numRows - 1, col);
      if (!areConnected(firstDevice, lastDevice)) {
        allColsRing = false;
      }
    }
  }

  std::vector<FabricConfig> perAxisConfig = {
      allRowsRing ? FabricConfig::FABRIC_1D_RING : FabricConfig::FABRIC_1D,
      allColsRing ? FabricConfig::FABRIC_1D_RING : FabricConfig::FABRIC_1D,
  };

  FabricConfig globalConfig = (allRowsRing && allColsRing)
                                  ? FabricConfig::FABRIC_1D_RING
                                  : FabricConfig::FABRIC_1D;

  return {globalConfig, perAxisConfig};
}

MeshFabricConfig computeFabricConfig(const ::tt::target::SystemDesc *systemDesc,
                                     const std::vector<uint32_t> &meshShape) {
  LOG_ASSERT(systemDesc != nullptr, "SystemDesc must not be null");

  std::vector<::tt::target::ChipChannel> chipChannels;
  auto fbChannels = systemDesc->chip_channels();
  if (fbChannels) {
    chipChannels.assign(fbChannels->begin(), fbChannels->end());
  }

  return computeFabricConfig(chipChannels, meshShape,
                             getMappedDeviceIds(meshShape));
}

} // namespace tt::runtime::common
