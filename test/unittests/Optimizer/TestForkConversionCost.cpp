// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ForkConversionCost.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace {

using ConversionKey = std::pair<unsigned, unsigned>;

TEST(ForkConversionCostTest, NoConversionsCostNothing) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> cost;
  EXPECT_EQ(cost.getBytes(), 0U);
  EXPECT_EQ(cost.getConversionCount(), 0U);
}

TEST(ForkConversionCostTest, RepeatedConsumersShareOneMaterialization) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> cost;
  for (unsigned consumer = 0; consumer < 3; ++consumer) {
    ASSERT_TRUE(cost.add({0, 1}, 8192, 16384));
  }
  EXPECT_EQ(cost.getBytes(), 24576U);
  EXPECT_EQ(cost.getConversionCount(), 1U);
}

TEST(ForkConversionCostTest, DifferentResultsDoNotShareMaterializations) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> cost;
  ASSERT_TRUE(cost.add({0, 1}, 8192, 8192));
  ASSERT_TRUE(cost.add({1, 1}, 8192, 8192));
  EXPECT_EQ(cost.getBytes(), 32768U);
  EXPECT_EQ(cost.getConversionCount(), 2U);
}

TEST(ForkConversionCostTest, DifferentTargetsDoNotShareMaterializations) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> cost;
  ASSERT_TRUE(cost.add({0, 1}, 8192, 8192));
  ASSERT_TRUE(cost.add({0, 2}, 8192, 16384));
  EXPECT_EQ(cost.getBytes(), 40960U);
  EXPECT_EQ(cost.getConversionCount(), 2U);
}

TEST(ForkConversionCostTest, BytesCanFavorMoreConversions) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> large;
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> small;
  ASSERT_TRUE(large.add({0, 1}, 65536, 65536));
  ASSERT_TRUE(small.add({0, 1}, 8192, 8192));
  ASSERT_TRUE(small.add({1, 1}, 8192, 8192));
  EXPECT_GT(small.getConversionCount(), large.getConversionCount());
  EXPECT_LT(small.getBytes(), large.getBytes());
}

TEST(ForkConversionCostTest, PhysicalFootprintIncludesTileAndShardPadding) {
  // A 33x33 bf16 tensor occupies four 32x32 tiles, not 33*33*2 bytes.
  EXPECT_EQ(mlir::tt::ttnn::getPhysicalBufferBytes(2048, std::array{2, 2},
                                                   std::array{1, 1}),
            8192U);
  EXPECT_EQ(mlir::tt::ttnn::getPhysicalBufferBytes(2048, std::array{1, 2},
                                                   std::array{2, 4}),
            32768U);
}

TEST(ForkConversionCostTest, EncodedTileSizeIncludesSharedExponents) {
  // BFP8 stores 64 shared exponent bytes in addition to 1024 payload bytes.
  EXPECT_EQ(mlir::tt::ttnn::getPhysicalBufferBytes(1088, std::array{1, 1},
                                                   std::array{1, 1}),
            1088U);
}

TEST(ForkConversionCostTest, UnknownOrEmptyDimensionsAreNotZeroCost) {
  EXPECT_FALSE(mlir::tt::ttnn::getPhysicalBufferBytes(2048, std::array{-1, 2},
                                                      std::array{1, 1}));
  EXPECT_FALSE(mlir::tt::ttnn::getPhysicalBufferBytes(2048, std::array{0, 2},
                                                      std::array{1, 1}));
  EXPECT_FALSE(mlir::tt::ttnn::getPhysicalBufferBytes(2048, std::array{1, 2},
                                                      std::array{0, 1}));
  EXPECT_FALSE(mlir::tt::ttnn::getPhysicalBufferBytes(0, std::array{1, 2},
                                                      std::array{1, 1}));
}

TEST(ForkConversionCostTest, MultiplicationOverflowIsNotCheap) {
  uint64_t limit = std::numeric_limits<uint64_t>::max();
  EXPECT_FALSE(mlir::tt::ttnn::getPhysicalBufferBytes(limit, std::array{2, 1},
                                                      std::array{1, 1}));
  EXPECT_FALSE(mlir::tt::ttnn::getPhysicalBufferBytes(limit, std::array{1, 1},
                                                      std::array{2, 1}));
  EXPECT_EQ(mlir::tt::ttnn::getPhysicalBufferBytes(limit, std::array{1, 1},
                                                   std::array{1, 1}),
            limit);
}

TEST(ForkConversionCostTest, SafeBuilderShapeIncludesNonTileMultiple) {
  EXPECT_TRUE(mlir::tt::ttnn::canRebuildForkLayout(std::array{33, 65},
                                                   std::array{8, 8}, true));
  EXPECT_TRUE(mlir::tt::ttnn::canRebuildForkLayout(std::array{1, 512, 512},
                                                   std::array{8, 8}, false));
}

TEST(ForkConversionCostTest, RejectsUnsafeBuilderIntermediates) {
  int64_t limit = std::numeric_limits<int64_t>::max();
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(
      std::array{limit, int64_t{32}}, std::array{1, 1}, true));
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(
      std::array{int64_t{1} << 32, int64_t{1} << 32}, std::array{1, 1}, false));
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(
      std::array{32, 32}, std::array{limit, limit}, false));
  // The existing L1-interleaved tile-count accumulator is int-sized.
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(
      std::array{int64_t{1} << 32, int64_t{1024}}, std::array{8, 8}, true));
}

TEST(ForkConversionCostTest, RejectsUnknownOrUnsupportedBuilderShape) {
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(std::array{-1, 32},
                                                    std::array{1, 1}, true));
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(std::array{0, 32},
                                                    std::array{1, 1}, false));
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(std::array{32, 32},
                                                    std::array{0, 1}, false));
  EXPECT_FALSE(mlir::tt::ttnn::canRebuildForkLayout(std::array{32},
                                                    std::array{1, 1}, false));
}

TEST(ForkConversionCostTest, ConversionOverflowLeavesCostUnchanged) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> cost;
  ASSERT_TRUE(cost.add({0, 1}, 32, 32));
  EXPECT_FALSE(cost.add({1, 1}, std::numeric_limits<uint64_t>::max(), 1));
  EXPECT_EQ(cost.getBytes(), 64U);
  EXPECT_EQ(cost.getConversionCount(), 1U);
}

TEST(ForkConversionCostTest, TotalOverflowLeavesCostUnchanged) {
  mlir::tt::ttnn::ForkConversionCost<ConversionKey> cost;
  uint64_t limit = std::numeric_limits<uint64_t>::max();
  ASSERT_TRUE(cost.add({0, 1}, limit - 32, 32));
  EXPECT_FALSE(cost.add({1, 1}, 1, 1));
  EXPECT_EQ(cost.getBytes(), limit);
  EXPECT_EQ(cost.getConversionCount(), 1U);
  // Reusing the existing conversion does not add any bytes, even at the limit.
  EXPECT_TRUE(cost.add({0, 1}, limit - 32, 32));
}

} // namespace
