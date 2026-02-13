// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"
#include <gtest/gtest.h>

using ::tt::runtime::FabricConfig;
using ::tt::runtime::MeshFabricConfig;
using ::tt::runtime::common::computeMeshFabricConfig;
using ::tt::target::ChipChannel;
using ::tt::target::Dim2d;

namespace {

ChipChannel makeChannel(uint32_t id0, uint32_t id1) {
  return ChipChannel(id0, Dim2d(0, 0), id1, Dim2d(0, 0));
}

} // namespace

// --- Single device: always DISABLED ---

TEST(ComputeMeshFabricConfig, SingleDevice1x1) {
  auto result = computeMeshFabricConfig({}, {1, 1}, {0});
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
  EXPECT_TRUE(result.perAxisConfig.empty());
}

// --- Two devices, no connection: both axes DISABLED ---

TEST(ComputeMeshFabricConfig, TwoDevices1x2NoConnection) {
  auto result = computeMeshFabricConfig({}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
}

// --- Two devices, connected: row ring, col DISABLED (single-element) ---

TEST(ComputeMeshFabricConfig, TwoDevices1x2WithWrap) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1)}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x1 column, connected: row DISABLED (single-element) ---

TEST(ComputeMeshFabricConfig, TwoDevices2x1WithWrap) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1)}, {2, 1}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, all wraparound: both axes ring ---

TEST(ComputeMeshFabricConfig, FourDevices2x2AllRing) {
  // Mesh layout (logical):
  //   0  1
  //   2  3
  // Row wraparound: 0<->1, 2<->3
  // Col wraparound: 0<->2, 1<->3
  auto result = computeMeshFabricConfig({makeChannel(0, 1), makeChannel(2, 3),
                                         makeChannel(0, 2), makeChannel(1, 3)},
                                        {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, only rows ring: col axis DISABLED ---

TEST(ComputeMeshFabricConfig, FourDevices2x2OnlyRowsRing) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1), makeChannel(2, 3)},
                                        {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, only cols ring: row axis DISABLED ---

TEST(ComputeMeshFabricConfig, FourDevices2x2OnlyColsRing) {
  auto result = computeMeshFabricConfig({makeChannel(0, 2), makeChannel(1, 3)},
                                        {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, no connections: both axes DISABLED ---

TEST(ComputeMeshFabricConfig, FourDevices2x2NoConnections) {
  auto result = computeMeshFabricConfig({}, {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
}

// --- Non-identity device ID mapping ---

TEST(ComputeMeshFabricConfig, NonIdentityMapping2x2) {
  // Physical IDs are remapped: logical [0,1,2,3] -> physical [3,1,2,0]
  // Mesh layout (physical):
  //   3  1
  //   2  0
  // Row wraparound needs: 3<->1, 2<->0
  // Col wraparound needs: 3<->2, 1<->0
  // Provide only row wraparound.
  auto result = computeMeshFabricConfig({makeChannel(1, 3), makeChannel(0, 2)},
                                        {2, 2}, {3, 1, 2, 0});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- Partial row: one row connected, other not → row axis DISABLED ---

TEST(ComputeMeshFabricConfig, FourDevices2x2PartialRow) {
  // Row 0: 0<->1 connected (ring). Row 1: 2<->3 NOT connected (disabled).
  // Axis = min(RING, DISABLED) = DISABLED.
  auto result =
      computeMeshFabricConfig({makeChannel(0, 1)}, {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
}

// --- Reversed channel order (id1 < id0 in ChipChannel) ---

TEST(ComputeMeshFabricConfig, ReversedChannelOrder) {
  auto result = computeMeshFabricConfig({makeChannel(1, 0)}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 1x4 ring: all adjacent + wraparound ---

TEST(ComputeMeshFabricConfig, FourDevices1x4Ring) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1), makeChannel(1, 2),
                                         makeChannel(2, 3), makeChannel(0, 3)},
                                        {1, 4}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 1x4 wraparound only (missing intermediate links): DISABLED ---

TEST(ComputeMeshFabricConfig, FourDevices1x4WrapOnly) {
  auto result =
      computeMeshFabricConfig({makeChannel(0, 3)}, {1, 4}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
}

// --- 1x4 linear: all adjacent, no wraparound ---

TEST(ComputeMeshFabricConfig, FourDevices1x4Linear) {
  auto result = computeMeshFabricConfig(
      {makeChannel(0, 1), makeChannel(1, 2), makeChannel(2, 3)}, {1, 4},
      {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 1x4 broken intermediate link: DISABLED ---

TEST(ComputeMeshFabricConfig, FourDevices1x4BrokenLink) {
  // 0<->1 ✓, 1<->2 ✗, 2<->3 ✓ → adjacent broken → DISABLED
  auto result = computeMeshFabricConfig({makeChannel(0, 1), makeChannel(2, 3)},
                                        {1, 4}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
}
