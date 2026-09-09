// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigHeuristic.h"

#include "llvm-gtest/gtest/gtest.h"

using namespace mlir::tt::ttnn;

TEST(Conv3dConfigHeuristicTest, ExactMatchRemapsFieldOrder) {
  // tt-metal _DEFAULT_BLOCKINGS entry:
  //   (192, 384, (3, 3, 3)): (64, 128, 1, 8, 4)
  // value tuple order = (C_in_block, C_out_block, T_out, H_out, W_out).
  std::optional<Conv3dBlocking> blocking =
      lookupConv3dBlocking(192, 384, {3, 3, 3});
  ASSERT_TRUE(blocking.has_value());
  EXPECT_EQ(blocking->tOutBlock, 1u);
  EXPECT_EQ(blocking->wOutBlock, 4u);
  EXPECT_EQ(blocking->hOutBlock, 8u);
  EXPECT_EQ(blocking->cOutBlock, 128u);
  EXPECT_EQ(blocking->cInBlock, 64u);
}

TEST(Conv3dConfigHeuristicTest, KernelIsPartOfKey) {
  // (192, 96, (1, 3, 3)) exists; (192, 96, (3, 3, 3)) does not.
  EXPECT_TRUE(lookupConv3dBlocking(192, 96, {1, 3, 3}).has_value());
  EXPECT_FALSE(lookupConv3dBlocking(192, 96, {3, 3, 3}).has_value());
}

TEST(Conv3dConfigHeuristicTest, MissReturnsNullopt) {
  // Not in the table.
  EXPECT_FALSE(lookupConv3dBlocking(128, 32, {3, 3, 3}).has_value());
  EXPECT_FALSE(lookupConv3dBlocking(7, 13, {3, 3, 3}).has_value());
}

TEST(Conv3dConfigHeuristicTest, WrongKernelRankIsMiss) {
  EXPECT_FALSE(lookupConv3dBlocking(192, 384, {3, 3}).has_value());
  EXPECT_FALSE(lookupConv3dBlocking(192, 384, {3, 3, 3, 3}).has_value());
}
