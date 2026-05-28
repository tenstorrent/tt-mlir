// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"

#include <set>

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/system_mesh.h"

namespace tt::runtime::common {

namespace {

// Maximum line length for which a wraparound link is promoted to a true
// FABRIC_1D_RING topology. Lines longer than this are kept at FABRIC_1D
// (linear) even when a physical wraparound link exists.
//
// Rationale (see https://github.com/tenstorrent/tt-mlir bug: 1x4 reduce_scatter
// hang): on Blackhole QuietBox2 a 4-chip "ring" reduce_scatter / all_gather
// deadlocks the ethernet data-movement workers, while the exact same op on a
// 2-chip axis (e.g. a 2x2 mesh) completes. Runtime evidence:
//   * 2x2 mesh: every collective is a 2-wide ring -> works (a 2-device ring is
//     degenerate: forward == backward over a single bidirectional link).
//   * 1x4 mesh: 4-wide ring reduce_scatter hangs on cores 14-3/14-2, even
//     though an identical OUTPUT-size reduce_scatter on the 2x2 (2-wide) axis
//     completes. The distinguishing factor is the ring WIDTH (>= 3 devices),
//     not the tensor size.
// The 4+-device ring reduce_scatter kernel (reduce_scatter_minimal_async ring
// program factory) relies on the bisection algorithm (N/2 hops, bidirectional)
// which is currently unreliable on this board. Falling back to the linear
// program factory (FABRIC_1D) avoids the hang at a modest latency cost
// (N-1 hops instead of N/2). 2-wide rings are unaffected, preserving the
// working 2x2 / per-axis-pair-of-2 paths.
constexpr size_t kMaxRingLineLength = 2;

// Classify a line of devices based on their connectivity:
//   DISABLED       — fewer than 2 devices, or any adjacent link is missing.
//   FABRIC_1D      — all adjacent pairs connected, no usable wraparound.
//   FABRIC_1D_RING — all adjacent pairs connected, last wraps to first, AND the
//                    line is short enough (<= kMaxRingLineLength) for the ring
//                    CCL kernels to be reliable.
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

  // A physical wraparound link exists. Only promote to a ring topology for
  // short lines; wider rings currently hang the reduce_scatter / all_gather
  // CCL kernels on Blackhole, so fall back to a linear topology.
  if (line.size() > kMaxRingLineLength) {
    LOG_WARNING("Line of ", line.size(),
                " devices has a wraparound link but exceeds the reliable ring "
                "size (",
                kMaxRingLineLength,
                "); falling back to FABRIC_1D (linear) topology to avoid CCL "
                "hangs.");
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
