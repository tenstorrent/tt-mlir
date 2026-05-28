// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm-gtest/gtest/gtest.h"

#include "gtest/gtest.h"
#include <set>

using namespace mlir::tt::ttnn;

namespace {
// Default-constructed Conv3dConfigAttr (all fields nullopt) used as the
// baseConfig for unit tests where no field is pinned.
Conv3dConfigAttr makeEmptyConv3dConfig(mlir::MLIRContext &context) {
  return Conv3dConfigAttr::get(
      &context,
      /*weights_dtype=*/std::optional<mlir::tt::ttcore::DataType>(),
      /*t_out_block=*/std::optional<uint32_t>(),
      /*w_out_block=*/std::optional<uint32_t>(),
      /*h_out_block=*/std::optional<uint32_t>(),
      /*c_out_block=*/std::optional<uint32_t>(),
      /*c_in_block=*/std::optional<uint32_t>(),
      /*compute_with_storage_grid_size=*/
      std::optional<mlir::tt::ttcore::GridAttr>());
}
} // namespace

class Conv3dConfigGeneratorTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  std::function<bool(const Conv3dConfigAttr &)> noFilter;

  void SetUp() override {
    context.loadDialect<TTNNDialect>();
    noFilter = [](const Conv3dConfigAttr &) { return false; };
  }
};

// An empty search space + empty base config means there is nothing to search.
// forEachConv3dConfig must return false and never invoke the callback.
TEST_F(Conv3dConfigGeneratorTest, EmptySearchSpaceIsDone) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  int count = 0;
  bool searched =
      forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                          noFilter, [&](Conv3dConfigAttr) { ++count; });
  EXPECT_FALSE(searched);
  EXPECT_EQ(count, 0);
}

// A single field with multiple candidate values produces one config per
// candidate, in order, then exhausts.
TEST_F(Conv3dConfigGeneratorTest, SingleFieldIteration) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  space.cInBlock = {32u, 64u, 96u};

  std::vector<Conv3dConfigAttr> configs;
  bool searched = forEachConv3dConfig(
      static_cast<Conv3dOp *>(nullptr), baseConfig, space, noFilter,
      [&](Conv3dConfigAttr cfg) { configs.push_back(cfg); });
  EXPECT_TRUE(searched);

  std::vector<uint32_t> seen;
  for (Conv3dConfigAttr config : configs) {
    ASSERT_TRUE(config.getCInBlock().has_value());
    seen.push_back(config.getCInBlock().value());
  }
  EXPECT_EQ(seen, (std::vector<uint32_t>{32u, 64u, 96u}));
}

// Multiple fields enumerate the full cartesian product. With 2x2 search
// values across two fields we expect exactly 4 distinct combinations.
TEST_F(Conv3dConfigGeneratorTest, CartesianProduct) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  space.cInBlock = {32u, 64u};
  space.hOutBlock = {1u, 2u};

  std::set<std::pair<uint32_t, uint32_t>> seen;
  forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                      noFilter, [&](Conv3dConfigAttr config) {
                        seen.insert({config.getCInBlock().value(),
                                     config.getHOutBlock().value()});
                      });
  EXPECT_EQ(seen.size(), 4u);
  EXPECT_TRUE(seen.count({32u, 1u}));
  EXPECT_TRUE(seen.count({32u, 2u}));
  EXPECT_TRUE(seen.count({64u, 1u}));
  EXPECT_TRUE(seen.count({64u, 2u}));
}

// A filterOutFn that rejects every candidate must produce zero configs but
// still report that the search itself was non-empty.
TEST_F(Conv3dConfigGeneratorTest, FilterOutRejectsAll) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  space.cInBlock = {32u, 64u};
  space.cOutBlock = {32u, 64u};

  int count = 0;
  bool searched = forEachConv3dConfig(
      static_cast<Conv3dOp *>(nullptr), baseConfig, space,
      [](const Conv3dConfigAttr &) { return true; },
      [&](Conv3dConfigAttr) { ++count; });
  EXPECT_TRUE(searched);
  EXPECT_EQ(count, 0);
}

// When baseConfig pins a field, that field is excluded from the search and
// its pinned value flows through unchanged on every emitted candidate.
TEST_F(Conv3dConfigGeneratorTest, BaseConfigPinningSkipsField) {
  // Pin c_in_block=96 on the base config. The search space still lists
  // c_in_block, but enumeration must ignore the search list and emit only
  // configs with c_in_block=96.
  Conv3dConfigAttr baseConfig = Conv3dConfigAttr::get(
      &context,
      /*weights_dtype=*/std::optional<mlir::tt::ttcore::DataType>(),
      /*t_out_block=*/std::optional<uint32_t>(),
      /*w_out_block=*/std::optional<uint32_t>(),
      /*h_out_block=*/std::optional<uint32_t>(),
      /*c_out_block=*/std::optional<uint32_t>(),
      /*c_in_block=*/std::optional<uint32_t>(96u),
      /*compute_with_storage_grid_size=*/
      std::optional<mlir::tt::ttcore::GridAttr>());

  Conv3dConfigSearchSpace space;
  space.cInBlock = {32u, 64u}; // Should be ignored: baseConfig pins it.
  space.hOutBlock = {1u, 2u};

  std::vector<Conv3dConfigAttr> configs;
  forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                      noFilter,
                      [&](Conv3dConfigAttr cfg) { configs.push_back(cfg); });

  std::set<uint32_t> cInBlockSeen;
  std::set<uint32_t> hOutBlockSeen;
  for (Conv3dConfigAttr config : configs) {
    ASSERT_TRUE(config.getCInBlock().has_value());
    cInBlockSeen.insert(config.getCInBlock().value());
    hOutBlockSeen.insert(config.getHOutBlock().value());
  }
  EXPECT_EQ(configs.size(), 2u); // hOutBlock=1, hOutBlock=2
  EXPECT_EQ(cInBlockSeen, (std::set<uint32_t>{96u}));
  EXPECT_EQ(hOutBlockSeen, (std::set<uint32_t>{1u, 2u}));
}

// A filterOutFn enforcing the empirical "h*w <= 256" rule must reject all
// candidates that violate it. With (h,w) drawn from {1,2,4,8,16,32} the
// product is bounded above by 32*32=1024, so several combinations get
// dropped.
TEST_F(Conv3dConfigGeneratorTest, FilterEnforcesInvariant) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  space.hOutBlock = {1u, 2u, 4u, 8u, 16u, 32u};
  space.wOutBlock = {1u, 2u, 4u, 8u, 16u, 32u};

  auto hwLeq256 = [](const Conv3dConfigAttr &cfg) {
    uint32_t h = cfg.getHOutBlock().value_or(1);
    uint32_t w = cfg.getWOutBlock().value_or(1);
    return h * w > 256;
  };

  int kept = 0;
  forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                      hwLeq256, [&](Conv3dConfigAttr config) {
                        uint32_t h = config.getHOutBlock().value();
                        uint32_t w = config.getWOutBlock().value();
                        EXPECT_LE(h * w, 256u);
                        ++kept;
                      });
  // 6x6 cartesian = 36 total. h*w > 256 rejects (16,32), (32,16), (32,32)
  // — three combinations — so 33 candidates pass through.
  EXPECT_EQ(kept, 33);
}

// On a realistic Wan-VAE-style shape (kernel 3x3x3, C_in=32, C_out=128,
// T_out=3, H_out=W_out=8) the realized candidate count after standard
// legality filters must stay tractable for LegalOpConfigAnalysis. Assert
// strict upper bound of 2000 — alarm if the generator ever explodes.
TEST_F(Conv3dConfigGeneratorTest, RealisticShapeCardinalityCap) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  // Block sizes derived from `models/tt_dit/tests/.../bruteforce_conv3d_sweep`:
  // C step 32, T divisors + strategic non-divisors, h/w in {1,2,4,8,16,32}.
  space.cInBlock = {32u};
  space.cOutBlock = {32u, 64u, 96u, 128u};
  space.tOutBlock = {1u, 3u};
  space.hOutBlock = {1u, 2u, 4u, 8u};
  space.wOutBlock = {1u, 2u, 4u, 8u};

  // Apply the legality filters from the sweep tool, condensed: divisibility
  // of output dims, h*w<=256.
  constexpr int64_t T_out = 3, H_out = 8, W_out = 8;
  constexpr int64_t C_in_aligned = 32, kT = 3, kH = 3, kW = 3;
  auto legality = [&](const Conv3dConfigAttr &cfg) {
    uint32_t cIn = cfg.getCInBlock().value();
    uint32_t cOut = cfg.getCOutBlock().value();
    uint32_t t = cfg.getTOutBlock().value();
    uint32_t h = cfg.getHOutBlock().value();
    uint32_t w = cfg.getWOutBlock().value();
    if (cIn == 0 || cOut == 0 || t == 0 || h == 0 || w == 0) {
      return true;
    }
    if ((kT * kH * kW * C_in_aligned) % cIn != 0) {
      return true;
    }
    if (T_out % t != 0 || H_out % h != 0 || W_out % w != 0) {
      return true;
    }
    if (h * w > 256) {
      return true;
    }
    return false;
  };

  int kept = 0;
  forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                      legality, [&](Conv3dConfigAttr) { ++kept; });
  EXPECT_GT(kept, 0); // must produce at least one viable candidate
  EXPECT_LE(kept, 2000)
      << "Generator cardinality exceeded the unit-test cap; tighten "
         "filterOutFn before wiring into LegalOpConfigAnalysis.";
}

// Exhaustion semantics: every candidate in the cartesian product is emitted
// exactly once.
TEST_F(Conv3dConfigGeneratorTest, ExhaustionSemantics) {
  Conv3dConfigAttr baseConfig = makeEmptyConv3dConfig(context);
  Conv3dConfigSearchSpace space;
  space.cInBlock = {32u, 64u};
  int count = 0;
  bool searched =
      forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                          noFilter, [&](Conv3dConfigAttr) { ++count; });
  EXPECT_TRUE(searched);
  EXPECT_EQ(count, 2);
}
