// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"

#include <set>

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/system_mesh.h"

namespace tt::runtime::common {

namespace {

// Classify a line of devices as FABRIC_1D_RING or FABRIC_1D. A line of devices
// is a ring if all adjacent pairs are connected AND the last device wraps
// around to the first. Falls back to FABRIC_1D otherwise.
FabricConfig
classifyLine(const std::vector<uint32_t> &line,
             const std::set<std::pair<uint32_t, uint32_t>> &connections) {
  if (line.size() < 2) {
    return FabricConfig::FABRIC_1D;
  }

  auto areConnected = [&connections](uint32_t id0, uint32_t id1) {
    if (id0 > id1) {
      std::swap(id0, id1);
    }
    return connections.count({id0, id1}) > 0;
  };

  for (size_t i = 0; i + 1 < line.size(); ++i) {
    if (!areConnected(line[i], line[i + 1])) {
      return FabricConfig::FABRIC_1D;
    }
  }

  if (!areConnected(line.front(), line.back())) {
    return FabricConfig::FABRIC_1D;
  }

  return FabricConfig::FABRIC_1D_RING;
}

} // namespace

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

  auto getDeviceAt = [&deviceIds, numCols](uint32_t row, uint32_t col) {
    int id = deviceIds[row * numCols + col];
    LOG_ASSERT(id >= 0, "Unmapped device at (", row, ", ", col, ")");
    return static_cast<uint32_t>(id);
  };

  // Classify each row and check if all rows agree on ring.
  bool allRowsRing = numCols > 1;
  for (uint32_t row = 0; row < numRows && allRowsRing; ++row) {
    std::vector<uint32_t> line(numCols);
    for (uint32_t col = 0; col < numCols; ++col) {
      line[col] = getDeviceAt(row, col);
    }
    allRowsRing =
        classifyLine(line, connections) == FabricConfig::FABRIC_1D_RING;
  }

  // Classify each column and check if all columns agree on ring.
  bool allColsRing = numRows > 1;
  for (uint32_t col = 0; col < numCols && allColsRing; ++col) {
    std::vector<uint32_t> line(numRows);
    for (uint32_t row = 0; row < numRows; ++row) {
      line[row] = getDeviceAt(row, col);
    }
    allColsRing =
        classifyLine(line, connections) == FabricConfig::FABRIC_1D_RING;
  }

  std::vector<FabricConfig> perAxisConfig = {
      allRowsRing ? FabricConfig::FABRIC_1D_RING : FabricConfig::FABRIC_1D,
      allColsRing ? FabricConfig::FABRIC_1D_RING : FabricConfig::FABRIC_1D,
  };

  FabricConfig globalConfig = (allRowsRing || allColsRing)
                                  ? FabricConfig::FABRIC_1D_RING
                                  : FabricConfig::FABRIC_1D;

  return {globalConfig, perAxisConfig};
}

MeshFabricConfig computeFabricConfig(const ::tt::target::SystemDesc *systemDesc,
                                     const std::vector<uint32_t> &meshShape) {
  LOG_ASSERT(systemDesc != nullptr, "SystemDesc must not be null");

  std::vector<::tt::target::ChipChannel> chipChannels;
  const auto *fbChannels = systemDesc->chip_channels();
  if (fbChannels) {
    chipChannels.reserve(fbChannels->size());
    for (const auto *channel : *fbChannels) {
      chipChannels.push_back(*channel);
    }
  }

  return computeFabricConfig(chipChannels, meshShape,
                             getMappedDeviceIds(meshShape));
}

} // namespace tt::runtime::common
