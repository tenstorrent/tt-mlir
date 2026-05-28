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
//
// Even though a physical wraparound link (0<->3) exists, a 4-wide ring
// reduce_scatter / all_gather currently hangs the CCL ethernet workers on
// Blackhole QuietBox2. classifyLine therefore caps ring promotion to lines of
// length <= kMaxRingLineLength (2), so a 1x4 axis with wraparound is reported
// as FABRIC_1D (linear) rather than FABRIC_1D_RING.
TEST(ComputeMeshFabricConfig, FourDevices1x4Ring) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1), makeChannel(1, 2),
                                         makeChannel(2, 3), makeChannel(0, 3)},
                                        {1, 4}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 1x3 ring downgraded to linear (wraparound but width > 2) ---

TEST(ComputeMeshFabricConfig, ThreeDevices1x3RingDowngradedToLinear) {
  auto result = computeMeshFabricConfig(
      {makeChannel(0, 1), makeChannel(1, 2), makeChannel(0, 2)}, {1, 3},
      {0, 1, 2});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 2x4: row axis (4-wide ring) downgraded, col axis (2-wide) stays ring ---

TEST(ComputeMeshFabricConfig, EightDevices2x4MixedRingDowngrade) {
  // Mesh layout (logical):
  //   0  1  2  3
  //   4  5  6  7
  // Row wraparound (4-wide): downgraded to FABRIC_1D.
  // Col wraparound (2-wide): kept as FABRIC_1D_RING.
  auto result = computeMeshFabricConfig(
      {// Row 0 ring: 0-1-2-3-0
       makeChannel(0, 1), makeChannel(1, 2), makeChannel(2, 3),
       makeChannel(0, 3),
       // Row 1 ring: 4-5-6-7-4
       makeChannel(4, 5), makeChannel(5, 6), makeChannel(6, 7),
       makeChannel(4, 7),
       // Columns (2-wide): 0-4, 1-5, 2-6, 3-7
       makeChannel(0, 4), makeChannel(1, 5), makeChannel(2, 6),
       makeChannel(3, 7)},
      {2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  // Row axis (4-wide) downgraded to linear.
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  // Col axis (2-wide) stays a ring.
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  // Best of the two -> RING.
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
