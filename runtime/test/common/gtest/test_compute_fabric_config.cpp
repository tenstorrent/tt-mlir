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

// --- Two devices, connected: the axis is ring-connected, but a 2-device wrap
// is degenerate, so the global fabric mode stays FABRIC_1D. ---

TEST(ComputeMeshFabricConfig, TwoDevices1x2WithWrap) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1)}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 2x1 column, connected: same degenerate 2-device rule. ---

TEST(ComputeMeshFabricConfig, TwoDevices2x1WithWrap) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1)}, {2, 1}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 2x2, all wraparound: both axes ring-connected, but both are only two
// devices wide (degenerate), so the global mode stays FABRIC_1D. ---

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
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 2x2, only rows ring: col axis DISABLED, rows degenerate → FABRIC_1D ---

TEST(ComputeMeshFabricConfig, FourDevices2x2OnlyRowsRing) {
  auto result = computeMeshFabricConfig({makeChannel(0, 1), makeChannel(2, 3)},
                                        {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- 2x2, only cols ring: row axis DISABLED, cols degenerate → FABRIC_1D ---

TEST(ComputeMeshFabricConfig, FourDevices2x2OnlyColsRing) {
  auto result = computeMeshFabricConfig({makeChannel(0, 2), makeChannel(1, 3)},
                                        {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::DISABLED);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
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
  // Provide only row wraparound (a degenerate 2-device row → FABRIC_1D global).
  auto result = computeMeshFabricConfig({makeChannel(1, 3), makeChannel(0, 2)},
                                        {2, 2}, {3, 1, 2, 0});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
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

// --- Reversed channel order (id1 < id0 in ChipChannel): still a degenerate
// 2-device wrap, so global stays FABRIC_1D. ---

TEST(ComputeMeshFabricConfig, ReversedChannelOrder) {
  auto result = computeMeshFabricConfig({makeChannel(1, 0)}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::DISABLED);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
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

// --- 2x4 llmbox-like: linear size-4 rows, degenerate size-2 columns.
// Regression for the "no forwarding direction (M0,D0)->(M0,D3)" fatal: the
// degenerate 2-device column axis must NOT push the global config to
// FABRIC_1D_RING, or metal (whose 1D axis_can_wrap is global) would treat the
// linear size-4 row axis as wrap-capable and fail to route the wraparound. ---

TEST(ComputeMeshFabricConfig, EightDevices2x4LinearRowsDegenerateCols) {
  // Layout:
  //   0 1 2 3
  //   4 5 6 7
  // Rows linear (no 0<->3 / 4<->7 wrap); columns connected (0<->4, 1<->5, ...).
  auto result = computeMeshFabricConfig(
      {makeChannel(0, 1), makeChannel(1, 2), makeChannel(2, 3),
       makeChannel(4, 5), makeChannel(5, 6), makeChannel(6, 7),
       makeChannel(0, 4), makeChannel(1, 5), makeChannel(2, 6),
       makeChannel(3, 7)},
      {2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- Genuine mixed mesh: 3x4 with ring rows (extent 4) and linear columns
// (extent 3). This is the case a 1D fabric cannot represent, so it maps to the
// per-axis 2D-torus-X mode (X == row/horizontal axis). ---

TEST(ComputeMeshFabricConfig, TwelveDevices3x4RingRowsLinearColsTorusX) {
  // Rows are genuine 4-device rings (adjacent + wrap); columns are linear.
  auto result = computeMeshFabricConfig(
      {// row rings (wrap col0<->col3 on each row)
       makeChannel(0, 1), makeChannel(1, 2), makeChannel(2, 3), makeChannel(0, 3),
       makeChannel(4, 5), makeChannel(5, 6), makeChannel(6, 7), makeChannel(4, 7),
       makeChannel(8, 9), makeChannel(9, 10), makeChannel(10, 11),
       makeChannel(8, 11),
       // columns linear (no row0<->row2 wrap)
       makeChannel(0, 4), makeChannel(4, 8), makeChannel(1, 5), makeChannel(5, 9),
       makeChannel(2, 6), makeChannel(6, 10), makeChannel(3, 7),
       makeChannel(7, 11)},
      {3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_2D_TORUS_X);
}

// --- Genuine mixed mesh: 4x3 with linear rows (extent 3) and ring columns
// (extent 4) → per-axis 2D-torus-Y mode (Y == col/vertical axis). ---

TEST(ComputeMeshFabricConfig, TwelveDevices4x3LinearRowsRingColsTorusY) {
  // Columns are genuine 4-device rings (adjacent + wrap); rows are linear.
  auto result = computeMeshFabricConfig(
      {// rows linear (no col0<->col2 wrap)
       makeChannel(0, 1), makeChannel(1, 2), makeChannel(3, 4), makeChannel(4, 5),
       makeChannel(6, 7), makeChannel(7, 8), makeChannel(9, 10),
       makeChannel(10, 11),
       // column rings (wrap row0<->row3 on each column)
       makeChannel(0, 3), makeChannel(3, 6), makeChannel(6, 9), makeChannel(0, 9),
       makeChannel(1, 4), makeChannel(4, 7), makeChannel(7, 10), makeChannel(1, 10),
       makeChannel(2, 5), makeChannel(5, 8), makeChannel(8, 11),
       makeChannel(2, 11)},
      {4, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_2D_TORUS_Y);
}
