// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "gtest/gtest.h"

using namespace mlir::tt;
using namespace mlir::tt::ttnn;

// Pins the DRAM-sharded shard geometry. computeShardParams is arithmetic, so
// no device is involved.
namespace {

// Wormhole-ish L1 budget: 0.95 * usable L1 (1499136).
constexpr int64_t kL1Available = 1424179;

// A decode-shaped 32x4096x4096 projection.
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

  // 128 N-tiles over 12 banks: 11 per bank, the last one padded.
  EXPECT_EQ(p->perCoreNCompute, 11);
  EXPECT_EQ(p->kTiles, 128);

  EXPECT_EQ(p->perCoreM, 1);
  // div_up(128 N-tiles, 64 cores).
  EXPECT_EQ(p->perCoreNStorage, 2);

  // in0_block_w divides K-per-core (128/8 = 16).
  EXPECT_GT(p->in0BlockW, 0);
  EXPECT_EQ((kK / 32) / kNumIn0Cores % p->in0BlockW, 0);
}

// A sub-tile batch still yields per_core_M == 1.
TEST(MatmulDRAMShardParams, SubTileBatchRoundsPerCoreMUp) {
  for (int64_t m : {1, 2, 15, 31, 32}) {
    auto p = computeShardParams(m, kK, kN, kWormholeBanks, kNumIn0Cores,
                                kWormholeCores, ttcore::DataType::BFP_BFloat8,
                                kL1Available);
    ASSERT_TRUE(p.has_value()) << "M=" << m;
    EXPECT_EQ(p->perCoreM, 1) << "M=" << m;
  }
}

// 8 banks divide 128 tiles evenly.
TEST(MatmulDRAMShardParams, BlackholeBankCountChangesShardWidth) {
  auto p = computeShardParams(kM, kK, kN, kBlackholeBanks, kNumIn0Cores,
                              kBlackholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  ASSERT_TRUE(p.has_value());

  EXPECT_EQ(p->perCoreNCompute, 16);
  EXPECT_EQ(p->numBanks, kBlackholeBanks);
  EXPECT_EQ(p->perCoreM, 1);
}

// bfp4 tiles are about half of bfp8, so anywhere bfp8 fits bfp4 must too, and
// some budget must separate them. Swept because at a generous budget both cap
// at K-per-core; N is wide so in1 dominates the budget. The sweep reaches
// 425000, where bfp8 falls under kMinBlockWidth and bfp4 still fits.
TEST(MatmulDRAMShardParams, Bfp4NeverFitsWorseThanBfp8) {
  constexpr int64_t kWideN = 8192;
  bool sawBfp4OnlyFit = false;

  for (int64_t l1 : {425000, 500000, 600000, 700000, 800000, 900000, 1000000,
                     1100000, static_cast<int>(kL1Available)}) {
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

// The fixed CBs alone exceed the budget.
TEST(MatmulDRAMShardParams, DeclinesWhenFixedCBsExceedBudget) {
  auto p = computeShardParams(kM, kK, kN, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              /*l1Available=*/100000);
  EXPECT_FALSE(p.has_value());
}

// A wide N inflates in1, so the search walks in0_block_w down from 16 to 8.
// Unconditional assertions: guarding on has_value would hide a fit regression.
TEST(MatmulDRAMShardParams, WideNWalksIn0BlockWDown) {
  auto p = computeShardParams(kM, kK, /*N=*/8192, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  ASSERT_TRUE(p.has_value());

  // 256 N-tiles over 12 banks.
  EXPECT_EQ(p->perCoreNCompute, 22);

  EXPECT_LT(p->in0BlockW, 16);
  EXPECT_EQ(p->in0BlockW, 8);

  EXPECT_EQ(p->perCoreM, 1);
  EXPECT_GT(p->in0BlockW, 0);
  EXPECT_EQ((kK / 32) / kNumIn0Cores % p->in0BlockW, 0);
}

// N=32768 leaves only in0_block_w=2 of 16, below the floor.
TEST(MatmulDRAMShardParams, WideNBlockCollapseDeclined) {
  auto p = computeShardParams(kM, kK, /*N=*/32768, kWormholeBanks, kNumIn0Cores,
                              kWormholeCores, ttcore::DataType::BFP_BFloat8,
                              kL1Available);
  EXPECT_FALSE(p.has_value());
}

// K-per-core 43 is prime, so the only widths are 43, which does not fit, and 1,
// which is below the floor: qwen_2_5_3b's down projection on 8 banks.
TEST(MatmulDRAMShardParams, PrimeKPerCoreCollapseDeclined) {
  constexpr int64_t kBlackholeL1Available = 1400832; // 0.95 * (1572864 - 98304)
  auto p = computeShardParams(
      kM, /*K=*/11008, /*N=*/2048, kBlackholeBanks, kNumIn0Cores,
      kBlackholeCores, ttcore::DataType::BFP_BFloat8, kBlackholeL1Available);
  EXPECT_FALSE(p.has_value());
}

} // namespace
