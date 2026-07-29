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

  // Choose the single fabric mode that metal is programmed with. It must
  // preserve the per-axis distinction: metal's 1D fabric derives wraparound
  // globally (get_axis_topology: axis_can_wrap == (config == FABRIC_1D_RING)),
  // so collapsing a mixed mesh to FABRIC_1D_RING makes every extent>2 axis --
  // including linear ones -- wrap-capable and produces "no forwarding
  // direction" fatals on the linear axis. Only the 2D-torus variants encode
  // wraparound per axis (X == row/horizontal axis, Y == col/vertical axis).
  //
  // A wraparound is only a genuine ring when it reaches a device distinct from
  // the ordinary neighbor, i.e. the axis extent is > 2. On a two-device axis
  // the wrap peer is the adjacent neighbor, so metal treats it as a mesh edge
  // (get_boundary_mode returns NONE for extent==2); programming a torus/ring
  // there is pointless and, for 2D torus, corrupts routing. So a degenerate
  // two-device "ring" is demoted to a linear axis here.
  const uint32_t rowAxisExtent = numCols; // row lines span the columns (X)
  const uint32_t colAxisExtent = numRows; // col lines span the rows (Y)

  const bool rowIsRing =
      rowAxisConfig == FabricConfig::FABRIC_1D_RING && rowAxisExtent > 2;
  const bool colIsRing =
      colAxisConfig == FabricConfig::FABRIC_1D_RING && colAxisExtent > 2;

  // An axis is a real second dimension for CCL when it is connected and spans
  // more than one device. Only then does a mixed (ring + linear) mesh require a
  // 2D-torus mode; otherwise the mesh is effectively 1D and we keep the leaner
  // 1D fabric mode that the common single-axis meshes rely on.
  const bool rowIsRealLinear = rowAxisConfig != FabricConfig::DISABLED &&
                               rowAxisExtent > 1 && !rowIsRing;
  const bool colIsRealLinear = colAxisConfig != FabricConfig::DISABLED &&
                               colAxisExtent > 1 && !colIsRing;

  FabricConfig globalConfig = FabricConfig::DISABLED;
  if (rowIsRing && colIsRealLinear) {
    // Ring on the X axis, genuine linear Y axis: keep Y linear.
    globalConfig = FabricConfig::FABRIC_2D_TORUS_X;
  } else if (colIsRing && rowIsRealLinear) {
    // Ring on the Y axis, genuine linear X axis.
    globalConfig = FabricConfig::FABRIC_2D_TORUS_Y;
  } else if (rowIsRing || colIsRing) {
    // A genuine ring with no competing linear axis (the other axis is also a
    // ring or is trivial/absent): the mesh is 1D-ring shaped.
    globalConfig = FabricConfig::FABRIC_1D_RING;
  } else if (rowAxisConfig != FabricConfig::DISABLED ||
             colAxisConfig != FabricConfig::DISABLED) {
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
