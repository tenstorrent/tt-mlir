// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "gtest/gtest.h"

using namespace mlir::tt;
using namespace mlir::tt::ttnn;

// computeShardParams is pure arithmetic over the matmul dims and the L1 budget,
// so it needs no MLIRContext and no device. These tests pin the DRAM-sharded
// shard geometry, which nothing else in the tree covers.
namespace {

// Wormhole-ish L1 budget: 0.95 * usable L1 (1499136).
constexpr int64_t kL1Available = 1424179;

// A decode-shaped 32x4096x4096 projection, the shape the DS path is built for.
constexpr int64_t kM = 32;
constexpr int64_t kK = 4096;
constexpr int64_t kN = 4096;

constexpr int64_t kWormholeBanks = 12;
constexpr int64_t kBlackholeBanks = 8;
constexpr int64_t kNumIn0Cores = 8;
constexpr int64_t kWormholeCores = 64;
constexpr int64_t kBlackholeCores = 110;

TEST(MatmulDRAMShardParams, WormholeBaseline) {
  auto p = computeShardParams(kM, kK, kN, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  ASSERT_TRUE(p.has_value());

  // N is padded up to a multiple of tile*banks = 32*12 = 384: 4096 -> 4224,
  // giving 4224/12 = 352 = 11 tiles of weight per bank.
  EXPECT_EQ(p->nPadded, 4224);
  EXPECT_EQ(p->shardW, 352);
  EXPECT_EQ(p->shardWTiles, 11);

  EXPECT_EQ(p->kTiles, 128);
  EXPECT_EQ(p->shardH, kK);

  // tt-metal requires M == per_core_M == 1 for the DS config.
  EXPECT_EQ(p->perCoreM, 1);
  // per_core_N is the *storage* split: div_up(128 N-tiles, 64 cores).
  EXPECT_EQ(p->perCoreN, 2);

  // in0_block_w must divide K-per-core (128/8 = 16) and fit the CBs.
  EXPECT_GT(p->in0BlockW, 0);
  EXPECT_EQ((kK / 32) / kNumIn0Cores % p->in0BlockW, 0);
}

// A sub-tile decode batch must still yield per_core_M == 1: tt-metal pads the
// activation up to one tile row. Truncating M/32 would give 0 here and build a
// degenerate config.
TEST(MatmulDRAMShardParams, SubTileBatchRoundsPerCoreMUp) {
  for (int64_t m : {1, 2, 15, 31, 32}) {
    auto p = computeShardParams(m, kK, kN, kWormholeBanks, kNumIn0Cores,
                                kWormholeCores, ttcore::DataType::BFP_BFloat8,
                                kL1Available);
    ASSERT_TRUE(p.has_value()) << "M=" << m;
    EXPECT_EQ(p->perCoreM, 1) << "M=" << m;
  }
}

// The bank count changes the weight shard width, which is why it must come from
// the device rather than a constant: on 8 banks 4096 needs no padding at all.
TEST(MatmulDRAMShardParams, BlackholeBankCountChangesShardWidth) {
  auto p = computeShardParams(kM, kK, kN, kBlackholeBanks, kNumIn0Cores,
                              kBlackholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  ASSERT_TRUE(p.has_value());

  // 32*8 = 256 already divides 4096, so nPadded == N.
  EXPECT_EQ(p->nPadded, kN);
  EXPECT_EQ(p->shardW, 512);
  EXPECT_EQ(p->shardWTiles, 16);
  EXPECT_EQ(p->numBanks, kBlackholeBanks);
  EXPECT_EQ(p->perCoreM, 1);
}

// bfp4 weight tiles are roughly half the bytes of bfp8, and the weight CB (in1)
// is the term that dominates the L1 budget, so anywhere bfp8 fits bfp4 must too
// — and there must be budgets where only bfp4 fits, or the dtype would not be
// affecting the decision at all.
//
// Keep this a sweep. At any single generous budget the chosen in0_block_w is
// capped at K-per-core for both dtypes, so comparing them there would assert
// nothing. N is the wide case so the in1 CB dominates the budget.
//
// kWideN is 8192 rather than something larger because kMinBlockWidthFraction
// declines a block width below half of K-per-core: at N=32768 neither dtype can
// reach in0_block_w=8 within any swept budget, so both return nullopt and the
// sweep stops discriminating. 8192 still puts in1 in charge of the budget while
// leaving in0_block_w in the accepted range.
TEST(MatmulDRAMShardParams, Bfp4NeverFitsWorseThanBfp8) {
  constexpr int64_t kWideN = 8192;
  bool sawBfp4OnlyFit = false;

  for (int64_t l1 : {600000, 700000, 800000, 900000, 1000000, 1100000,
                     static_cast<int>(kL1Available)}) {
    auto bfp8 =
        computeShardParams(kM, kK, kWideN, kWormholeBanks, kNumIn0Cores,
                           kWormholeCores, ttcore::DataType::BFP_BFloat8, l1);
    auto bfp4 =
        computeShardParams(kM, kK, kWideN, kWormholeBanks, kNumIn0Cores,
                           kWormholeCores, ttcore::DataType::BFP_BFloat4, l1);
    if (bfp8.has_value()) {
      EXPECT_TRUE(bfp4.has_value()) << "bfp8 fit but bfp4 did not, l1=" << l1;
      EXPECT_EQ(bfp4->weightDataType, ttcore::DataType::BFP_BFloat4);
    }
    if (bfp4.has_value() && !bfp8.has_value()) {
      sawBfp4OnlyFit = true;
    }
  }

  EXPECT_TRUE(sawBfp4OnlyFit)
      << "no swept budget separated bfp4 from bfp8, so this test is not "
         "exercising the weight-dtype term of the CB budget any more";
}

// The fixed CBs (out + interm0) alone exceed a tiny budget, so no in0_block_w
// can rescue it and the DS path declines rather than emitting a config that
// would not fit L1.
TEST(MatmulDRAMShardParams, DeclinesWhenFixedCBsExceedBudget) {
  auto p = computeShardParams(kM, kK, kN, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              /*l1Available=*/100000);
  EXPECT_FALSE(p.has_value());
}

// A wide N inflates the per-bank weight shard and therefore the in1 CB, forcing
// the search to walk in0_block_w down from K-per-core (16) to something that
// fits. Keep these assertions unconditional: guarding them behind
// p.has_value() would let the test pass silently the moment the geometry stops
// fitting, which is the regression worth catching.
//
// N=8192 walks down exactly one divisor, 16 -> 8, which is the widest reduction
// kMinBlockWidthFraction still accepts. Anything wider is declined instead --
// see WideNBlockCollapseDeclined.
TEST(MatmulDRAMShardParams, WideNWalksIn0BlockWDown) {
  auto p = computeShardParams(kM, kK, /*N=*/8192, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  ASSERT_TRUE(p.has_value());

  // 8192 pads to a multiple of 32*12=384 -> 8448, /12 = 704 = 22 tiles/bank.
  EXPECT_EQ(p->nPadded, 8448);
  EXPECT_EQ(p->shardWTiles, 22);

  // The search starts at K-per-core (128/8 = 16) and must come down to fit.
  EXPECT_LT(p->in0BlockW, 16);
  EXPECT_EQ(p->in0BlockW, 8);

  // Invariants tt-metal checks regardless of where the search lands.
  EXPECT_EQ(p->perCoreM, 1);
  EXPECT_GT(p->in0BlockW, 0);
  EXPECT_EQ((kK / 32) / kNumIn0Cores % p->in0BlockW, 0);
  EXPECT_EQ(p->nPadded % (32 * kWormholeBanks), 0);
}

// Past half of K-per-core the DS path declines instead of emitting the config.
// N=32768 leaves room only for in0_block_w=2 out of K-per-core=16, i.e. 64
// block iterations instead of 8.
TEST(MatmulDRAMShardParams, WideNBlockCollapseDeclined) {
  auto p = computeShardParams(kM, kK, /*N=*/32768, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  EXPECT_FALSE(p.has_value());
}

// The shape that motivated the guard: qwen_2_5_3b's down-projection on an
// 8-bank part. K=11008 is 344 tiles, so K-per-core is the prime 43 and the only
// legal block widths are 43 and 1. 43 does not fit, so the search would emit
// in0_block_w=1 -- 344 serialized blocks instead of 8.
TEST(MatmulDRAMShardParams, PrimeKPerCoreCollapseDeclined) {
  constexpr int64_t kBlackholeL1Available = 1400832; // 0.95 * (1572864 - 98304)
  auto p = computeShardParams(
      kM, /*K=*/11008, /*N=*/2048, kBlackholeBanks, kNumIn0Cores,
      kBlackholeCores, ttcore::DataType::BFP_BFloat8, kBlackholeL1Available);
  EXPECT_FALSE(p.has_value());
}

} // namespace
