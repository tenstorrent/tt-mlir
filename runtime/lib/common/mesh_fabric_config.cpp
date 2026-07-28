// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"

#include <set>

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/system_mesh.h"

namespace tt::runtime::common {

namespace {

// Classify a line of devices based on their connectivity:
//   DISABLED       — fewer than 2 devices, or any adjacent link is missing.
//   FABRIC_1D      — all adjacent pairs connected, no wraparound.
//   FABRIC_1D_RING — all adjacent pairs connected AND last wraps to first.
//
// If any adjacent link is broken the line cannot form even a linear topology.
FabricConfig
classifyLine(const std::vector<uint32_t> &line,
             const std::set<std::pair<uint32_t, uint32_t>> &connections) {
  if (line.size() < 2) {
    return FabricConfig::DISABLED;
  }

  auto areConnected = [&connections](uint32_t id0, uint32_t id1) {
    if (id0 > id1) {
      std::swap(id0, id1);
    }
    return connections.count({id0, id1}) > 0;
  };

  for (size_t i = 0; i + 1 < line.size(); ++i) {
    if (!areConnected(line[i], line[i + 1])) {
      LOG_WARNING("Devices ", line[i], " and ", line[i + 1],
                  " are not connected, disabling line.");
      return FabricConfig::DISABLED;
    }
  }

  if (!areConnected(line.front(), line.back())) {
    return FabricConfig::FABRIC_1D;
  }

  return FabricConfig::FABRIC_1D_RING;
}

// Classify an axis by taking the worst classification across all its lines.
// A single broken line downgrades the entire axis:
//   - Any line DISABLED  -> axis is DISABLED
//   - Any line FABRIC_1D -> axis is at most FABRIC_1D
//   - All lines RING     -> axis is FABRIC_1D_RING
FabricConfig
classifyAxis(const std::vector<std::vector<uint32_t>> &lines,
             const std::set<std::pair<uint32_t, uint32_t>> &connections) {
  bool allRing = true;
  for (const auto &line : lines) {
    FabricConfig lineConfig = classifyLine(line, connections);
    if (lineConfig == FabricConfig::DISABLED) {
      return FabricConfig::DISABLED;
    }
    if (lineConfig == FabricConfig::FABRIC_1D) {
      allRing = false;
    }
  }
  return allRing ? FabricConfig::FABRIC_1D_RING : FabricConfig::FABRIC_1D;
}

} // namespace

MeshFabricConfig computeMeshFabricConfig(
    const std::vector<::tt::target::ChipChannel> &chipChannels,
    const std::vector<uint32_t> &meshShape, const std::vector<int> &deviceIds) {
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

  // Build row-axis and column-axis lines of devices.
  std::vector<std::vector<uint32_t>> rowLines(numRows,
                                              std::vector<uint32_t>(numCols));
  std::vector<std::vector<uint32_t>> colLines(numCols,
                                              std::vector<uint32_t>(numRows));
  for (uint32_t row = 0; row < numRows; ++row) {
    for (uint32_t col = 0; col < numCols; ++col) {
      int id = deviceIds[row * numCols + col];
      LOG_ASSERT(id >= 0, "Unmapped device at (", row, ", ", col, ")");
      auto device = static_cast<uint32_t>(id);
      rowLines[row][col] = device;
      colLines[col][row] = device;
    }
  }

  FabricConfig rowAxisConfig = classifyAxis(rowLines, connections);
  FabricConfig colAxisConfig = classifyAxis(colLines, connections);

  std::vector<FabricConfig> perAxisConfig = {rowAxisConfig, colAxisConfig};

  // A wraparound on a 2-element axis is degenerate: front and back are the same
  // adjacent pair, so classifyLine reports RING for any connected pair. Only an
  // axis longer than 2 can carry a wrap edge distinct from its line edges.
  //
  // Axis-to-torus-dimension mapping follows tt-metal's get_valid_connections:
  // TORUS_X wraps mesh_shape[1] (E/W), which is the axis a row line runs along;
  // TORUS_Y wraps mesh_shape[0] (N/S), which is the axis a column line runs
  // along.
  bool rowAxisIsTorus =
      numCols > 2 && rowAxisConfig == FabricConfig::FABRIC_1D_RING;
  bool colAxisIsTorus =
      numRows > 2 && colAxisConfig == FabricConfig::FABRIC_1D_RING;

  // On a genuinely 2D mesh whose long axis wraps, request a 2D torus config
  // rather than FABRIC_1D_RING. tt-metal's get_fabric_type() maps
  // FABRIC_1D_RING to FabricType::MESH on anything that is not a UBB galaxy,
  // which drops the wrap edges from intra-mesh connectivity while collectives
  // still run a ring algorithm -- they then ask fabric to forward across the
  // wrap and abort in fabric.cpp with "Could not find any forwarding
  // direction". FABRIC_2D_TORUS_{X,Y,XY} map to the matching FabricType, so the
  // wrap edges are actually built. See docs/fabric-1d-ring-torus-mismatch.md.
  //
  // The mesh graph descriptor must also declare the wrapping dimension RING, or
  // tt-metal rejects the request in requires_more_connectivity().
  if (numRows > 1 && numCols > 1 && (rowAxisIsTorus || colAxisIsTorus)) {
    FabricConfig torusConfig = FabricConfig::FABRIC_2D_TORUS_XY;
    if (!colAxisIsTorus) {
      torusConfig = FabricConfig::FABRIC_2D_TORUS_X;
    } else if (!rowAxisIsTorus) {
      torusConfig = FabricConfig::FABRIC_2D_TORUS_Y;
    }
    return {torusConfig, perAxisConfig};
  }

  // Global config is the best of the two axes: if at least one axis supports
  // a topology, the fabric can use it.
  FabricConfig globalConfig = FabricConfig::DISABLED;
  if (rowAxisConfig == FabricConfig::FABRIC_1D_RING ||
      colAxisConfig == FabricConfig::FABRIC_1D_RING) {
    globalConfig = FabricConfig::FABRIC_1D_RING;
  } else if (rowAxisConfig == FabricConfig::FABRIC_1D ||
             colAxisConfig == FabricConfig::FABRIC_1D) {
    globalConfig = FabricConfig::FABRIC_1D;
  }

  return {globalConfig, perAxisConfig};
}

MeshFabricConfig
computeMeshFabricConfig(const ::tt::target::SystemDesc *systemDesc,
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

  return computeMeshFabricConfig(chipChannels, meshShape,
                                 getMappedDeviceIds(meshShape));
}

} // namespace tt::runtime::common
