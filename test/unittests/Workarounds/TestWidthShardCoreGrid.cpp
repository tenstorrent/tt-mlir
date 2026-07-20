// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormWidthShardInputRewritePattern.h"

#include <gtest/gtest.h>

namespace mlir::tt::ttnn::workarounds::decomposition {
namespace {

// The Wormhole worker grid is 8x8; every case below uses it.
constexpr int64_t kWorkerGridH = 8;
constexpr int64_t kWorkerGridW = 8;

WidthShardCoreGrid choose(int64_t numWidthTiles) {
  return chooseWidthShardCoreGrid(numWidthTiles, kWorkerGridH, kWorkerGridW);
}

// The regression this guards: Kimi-K2.6 b128 galaxy decode. 896 hidden / 32 =
// 28 width tiles. The old row-major placement put 28 cores on an 8-wide grid,
// yielding a ragged set whose bounding box was 8x4 (four empty cores) and
// corrupting the fused kernel. The shard must instead be a solid rectangle
// matching tt-metal's validated decode config: 4 wide x 7 tall.
TEST(WidthShardCoreGrid, DeepSeek896DecodeIsSolid4x7) {
  WidthShardCoreGrid grid = choose(/*numWidthTiles=*/28);
  EXPECT_EQ(grid.numCores, 28);
  EXPECT_EQ(grid.gridW, 4);
  EXPECT_EQ(grid.gridH, 7);
  // The essential invariant: the rectangle is solid, so its bounding box has
  // no empty cores.
  EXPECT_EQ(grid.gridW * grid.gridH, grid.numCores);
}

// For any selection, the placement must be a solid rectangle that fits the
// worker grid -- this is the property the fused kernel depends on.
TEST(WidthShardCoreGrid, PlacementIsAlwaysSolidAndFits) {
  for (int64_t tiles = 1; tiles <= 64; ++tiles) {
    WidthShardCoreGrid grid = choose(tiles);
    EXPECT_EQ(grid.gridW * grid.gridH, grid.numCores)
        << "tiles=" << tiles << " produced a ragged (non-rectangular) grid";
    EXPECT_LE(grid.gridW, kWorkerGridW) << "tiles=" << tiles;
    EXPECT_LE(grid.gridH, kWorkerGridH) << "tiles=" << tiles;
    EXPECT_GE(grid.numCores, 1) << "tiles=" << tiles;
    // The chosen core count must evenly divide the tiles so each core owns an
    // equal width slice.
    EXPECT_EQ(tiles % grid.numCores, 0) << "tiles=" << tiles;
  }
}

// The selection prefers the TALLEST rectangle (to land 28 -> 4x7). For counts
// that fill full rows this shows up as the transpose of the naive row-major
// shape -- still a solid rectangle, just oriented tall.
TEST(WidthShardCoreGrid, PrefersTallestRectangle) {
  // 16 = 2 wide x 8 tall (height maxed at the worker grid height first).
  WidthShardCoreGrid g16 = choose(/*numWidthTiles=*/16);
  EXPECT_EQ(g16.numCores, 16);
  EXPECT_EQ(g16.gridW, 2);
  EXPECT_EQ(g16.gridH, 8);

  // 24 = 3 wide x 8 tall.
  WidthShardCoreGrid g24 = choose(/*numWidthTiles=*/24);
  EXPECT_EQ(g24.numCores, 24);
  EXPECT_EQ(g24.gridW, 3);
  EXPECT_EQ(g24.gridH, 8);
}

// A count <= the worker grid height becomes a single tall column.
TEST(WidthShardCoreGrid, SmallCountIsSingleColumn) {
  WidthShardCoreGrid grid = choose(/*numWidthTiles=*/4);
  EXPECT_EQ(grid.numCores, 4);
  EXPECT_EQ(grid.gridW, 1);
  EXPECT_EQ(grid.gridH, 4);
}

// A prime tile count larger than the grid admits no multi-core rectangle, so
// it must fall back to a single core rather than a ragged placement.
TEST(WidthShardCoreGrid, PrimeLargerThanGridFallsBackToSingleCore) {
  WidthShardCoreGrid grid = choose(/*numWidthTiles=*/11);
  EXPECT_EQ(grid.numCores, 1);
  EXPECT_EQ(grid.gridW, 1);
  EXPECT_EQ(grid.gridH, 1);
}

} // namespace
} // namespace mlir::tt::ttnn::workarounds::decomposition
