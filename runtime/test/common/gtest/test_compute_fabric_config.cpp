// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/mesh_fabric_config.h"
#include <gtest/gtest.h>

using ::tt::runtime::FabricConfig;
using ::tt::runtime::MeshFabricConfig;
using ::tt::runtime::common::computeFabricConfig;
using ::tt::target::ChipChannel;
using ::tt::target::Dim2d;

namespace {

ChipChannel makeChannel(uint32_t id0, uint32_t id1) {
  return ChipChannel(id0, Dim2d(0, 0), id1, Dim2d(0, 0));
}

} // namespace

// --- Single device: always DISABLED ---

TEST(ComputeFabricConfig, SingleDevice1x1) {
  auto result = computeFabricConfig({}, {1, 1}, {0});
  EXPECT_EQ(result.globalConfig, FabricConfig::DISABLED);
  EXPECT_TRUE(result.perAxisConfig.empty());
}

// --- Two devices, no wraparound: linear ---

TEST(ComputeFabricConfig, TwoDevices1x2NoWrap) {
  auto result = computeFabricConfig({}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- Two devices, with wraparound: ring ---

TEST(ComputeFabricConfig, TwoDevices1x2WithWrap) {
  auto result = computeFabricConfig({makeChannel(0, 1)}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x1 column, with col wraparound ---

TEST(ComputeFabricConfig, TwoDevices2x1WithWrap) {
  auto result = computeFabricConfig({makeChannel(0, 1)}, {2, 1}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, all wraparound: both axes ring ---

TEST(ComputeFabricConfig, FourDevices2x2AllRing) {
  // Mesh layout (logical):
  //   0  1
  //   2  3
  // Row wraparound: 0<->1, 2<->3
  // Col wraparound: 0<->2, 1<->3
  auto result = computeFabricConfig({makeChannel(0, 1), makeChannel(2, 3),
                                     makeChannel(0, 2), makeChannel(1, 3)},
                                    {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, only rows ring ---

TEST(ComputeFabricConfig, FourDevices2x2OnlyRowsRing) {
  auto result = computeFabricConfig({makeChannel(0, 1), makeChannel(2, 3)},
                                    {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, only cols ring ---

TEST(ComputeFabricConfig, FourDevices2x2OnlyColsRing) {
  auto result = computeFabricConfig({makeChannel(0, 2), makeChannel(1, 3)},
                                    {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 2x2, no connections at all: linear ---

TEST(ComputeFabricConfig, FourDevices2x2NoConnections) {
  auto result = computeFabricConfig({}, {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- Non-identity device ID mapping ---

TEST(ComputeFabricConfig, NonIdentityMapping2x2) {
  // Physical IDs are remapped: logical [0,1,2,3] -> physical [3,1,2,0]
  // Mesh layout (physical):
  //   3  1
  //   2  0
  // Row wraparound needs: 3<->1, 2<->0
  // Col wraparound needs: 3<->2, 1<->0
  // Provide only row wraparound.
  auto result = computeFabricConfig({makeChannel(1, 3), makeChannel(0, 2)},
                                    {2, 2}, {3, 1, 2, 0});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- Partial row ring: one row has wrap, other doesn't ---

TEST(ComputeFabricConfig, FourDevices2x2PartialRowRing) {
  // Row 0: 0<->1 connected. Row 1: 2<->3 NOT connected.
  auto result = computeFabricConfig({makeChannel(0, 1)}, {2, 2}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}

// --- Reversed channel order (id1 < id0 in ChipChannel) ---

TEST(ComputeFabricConfig, ReversedChannelOrder) {
  auto result = computeFabricConfig({makeChannel(1, 0)}, {1, 2}, {0, 1});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
}

// --- 1x4 ring ---

TEST(ComputeFabricConfig, FourDevices1x4Ring) {
  auto result = computeFabricConfig({makeChannel(0, 3)}, {1, 4}, {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D_RING);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D_RING);
}

// --- 1x4 linear (no wraparound) ---

TEST(ComputeFabricConfig, FourDevices1x4Linear) {
  auto result = computeFabricConfig(
      {makeChannel(0, 1), makeChannel(1, 2), makeChannel(2, 3)}, {1, 4},
      {0, 1, 2, 3});

  ASSERT_EQ(result.perAxisConfig.size(), 2u);
  EXPECT_EQ(result.perAxisConfig[0], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.perAxisConfig[1], FabricConfig::FABRIC_1D);
  EXPECT_EQ(result.globalConfig, FabricConfig::FABRIC_1D);
}
