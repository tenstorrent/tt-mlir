// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"

#include <numeric>
#include <set>
#include <utility>

#include "tt/runtime/detail/common/system_mesh.h"
#include "ttmlir/Target/Common/types_generated.h"

namespace tt::runtime::common {

MeshFabricConfig computeFabricConfig(const ::tt::target::SystemDesc *systemDesc,
                                     const std::vector<uint32_t> &meshShape) {
  uint32_t totalDevices = std::accumulate(meshShape.begin(), meshShape.end(),
                                          1u, std::multiplies<uint32_t>());

  if (totalDevices <= 1) {
    return {FabricConfig::DISABLED, {}};
  }

  std::set<std::pair<uint32_t, uint32_t>> connections;
  auto chipChannels = systemDesc->chip_channels();
  if (chipChannels) {
    for (size_t i = 0; i < chipChannels->size(); ++i) {
      const auto *channel = chipChannels->Get(i);
      uint32_t id0 = channel->device_id0();
      uint32_t id1 = channel->device_id1();
      if (id0 > id1) {
        std::swap(id0, id1);
      }
      connections.insert({id0, id1});
    }
  }

  uint32_t numRows = meshShape.size() >= 1 ? meshShape[0] : 1;
  uint32_t numCols = meshShape.size() >= 2 ? meshShape[1] : totalDevices;

  std::vector<int> deviceIds = getMappedDeviceIds(meshShape);

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
  // connect.
  bool allRowsRing = true;
  for (uint32_t row = 0; row < numRows && allRowsRing; ++row) {
    if (numCols <= 1) {
      continue;
    }
    uint32_t firstDevice = getDeviceAt(row, 0);
    uint32_t lastDevice = getDeviceAt(row, numCols - 1);
    if (!areConnected(firstDevice, lastDevice)) {
      allRowsRing = false;
    }
  }

  // Check column wraparound: for each column, check if first and last connect.
  bool allColsRing = true;
  for (uint32_t col = 0; col < numCols && allColsRing; ++col) {
    if (numRows <= 1) {
      continue;
    }
    uint32_t firstDevice = getDeviceAt(0, col);
    uint32_t lastDevice = getDeviceAt(numRows - 1, col);
    if (!areConnected(firstDevice, lastDevice)) {
      allColsRing = false;
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

} // namespace tt::runtime::common
