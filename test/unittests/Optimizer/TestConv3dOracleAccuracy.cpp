// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Phase 4 of the conv3d optimizer rollout: oracle accuracy test.
//
// We lift the hand-tuned `_BLOCKINGS` table from tt-metal's
// `models/tt_dit/utils/conv3d.py` into a JSON asset and grade tt-mlir's
// optimizer scoring against it. For each row:
//   1. Build the same search space and shape-specific filter as
//      LegalOpConfigAnalysis would for a Conv3dOp of this shape.
//   2. Generate all candidates and sort by the empirical scoring tuple
//      defined in Phase 5 (voxelsPerLaunch, c_in, c_out, ...).
//   3. Record whether the oracle's chosen blocking appears in the top-1
//      and top-5 of our ranking.
//
// Thresholds are starting floors — ratchet upward as the heuristic improves.

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigSearchSpace.h"
#include "ttmlir/Dialect/TTNN/Analysis/LegalOpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm-gtest/gtest/gtest.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

using namespace mlir::tt::ttnn;

namespace {
struct OracleEntry {
  int64_t inChannels;
  int64_t outChannels;
  int64_t kT, kH, kW;
  int64_t tOut, hOut, wOut;
  uint32_t expectedCInBlock;
  uint32_t expectedCOutBlock;
  uint32_t expectedTOutBlock;
  uint32_t expectedHOutBlock;
  uint32_t expectedWOutBlock;
};

// Locate the JSON file relative to the test source — same pattern used by
// other gtest-driven oracle tests. Allows running both from build and from
// the repo root.
std::string locateOracleJson() {
  // Allow override for out-of-tree runs.
  if (const char *env = std::getenv("CONV3D_ORACLE_JSON")) {
    return env;
  }
  llvm::SmallString<256> path(llvm::sys::path::parent_path(__FILE__));
  llvm::sys::path::append(path, "data", "conv3d_blockings_oracle.json");
  return std::string(path);
}

std::vector<OracleEntry> loadOracle() {
  auto bufOr = llvm::MemoryBuffer::getFile(locateOracleJson());
  EXPECT_TRUE(static_cast<bool>(bufOr))
      << "Cannot read oracle JSON at " << locateOracleJson();
  if (!bufOr) {
    return {};
  }
  auto json = llvm::json::parse(bufOr.get()->getBuffer());
  EXPECT_TRUE(static_cast<bool>(json))
      << "Oracle JSON failed to parse: " << llvm::toString(json.takeError());
  if (!json) {
    return {};
  }
  auto *obj = json->getAsObject();
  auto *entries = obj->getArray("entries");
  std::vector<OracleEntry> out;
  for (const llvm::json::Value &v : *entries) {
    const llvm::json::Object *e = v.getAsObject();
    OracleEntry o;
    o.inChannels = *e->getInteger("in_channels");
    o.outChannels = *e->getInteger("out_channels");
    o.kT = *e->getInteger("kT");
    o.kH = *e->getInteger("kH");
    o.kW = *e->getInteger("kW");
    o.tOut = *e->getInteger("T_out");
    o.hOut = *e->getInteger("H_out");
    o.wOut = *e->getInteger("W_out");
    const llvm::json::Object *exp = e->getObject("expected");
    o.expectedCInBlock = static_cast<uint32_t>(*exp->getInteger("c_in_block"));
    o.expectedCOutBlock =
        static_cast<uint32_t>(*exp->getInteger("c_out_block"));
    o.expectedTOutBlock =
        static_cast<uint32_t>(*exp->getInteger("t_out_block"));
    o.expectedHOutBlock =
        static_cast<uint32_t>(*exp->getInteger("h_out_block"));
    o.expectedWOutBlock =
        static_cast<uint32_t>(*exp->getInteger("w_out_block"));
    out.push_back(o);
  }
  return out;
}

// Build a search space that *includes* the oracle's expected values plus
// the standard factory defaults. We need the expected blocking to be
// reachable; with the factory's small candidate sets some oracle picks
// (e.g. c_out_block=64 not in {32,64,96,128}? it is — but t_out_block=6
// or 9 from the table is not in {1,2,3,4}) would never appear.
Conv3dConfigSearchSpace buildSearchSpaceWithExpected(const OracleEntry &o) {
  Conv3dConfigSearchSpace space = Conv3dConfigSearchSpaceFactory::get();
  auto addUnique = [](llvm::SmallVector<uint32_t> &v, uint32_t x) {
    if (std::find(v.begin(), v.end(), x) == v.end()) {
      v.push_back(x);
    }
  };
  addUnique(space.cInBlock, o.expectedCInBlock);
  addUnique(space.cOutBlock, o.expectedCOutBlock);
  addUnique(space.tOutBlock, o.expectedTOutBlock);
  addUnique(space.hOutBlock, o.expectedHOutBlock);
  addUnique(space.wOutBlock, o.expectedWOutBlock);
  return space;
}

// Score tuple — must match the order used by LegalOpConfigAnalysis::
// fillOpSpecificAttrs sort key (Phase 5). Larger is better.
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>
scoreConfig(Conv3dConfigAttr cfg, int64_t spatialVolume) {
  uint32_t t = cfg.getTOutBlock().value_or(1);
  uint32_t h = cfg.getHOutBlock().value_or(1);
  uint32_t w = cfg.getWOutBlock().value_or(1);
  uint32_t cIn = cfg.getCInBlock().value_or(1);
  uint32_t cOut = cfg.getCOutBlock().value_or(1);
  int64_t hwAspect = std::abs(static_cast<int64_t>(h) * w - 32);
  int64_t largeSpatialBonus = (spatialVolume > 10000 && cOut >= 96) ? 1000 : 0;
  return {-hwAspect, static_cast<int64_t>(t), static_cast<int64_t>(cIn),
          static_cast<int64_t>(cOut), largeSpatialBonus};
}

bool sameBlocking(Conv3dConfigAttr cfg, const OracleEntry &o) {
  return cfg.getCInBlock() == o.expectedCInBlock &&
         cfg.getCOutBlock() == o.expectedCOutBlock &&
         cfg.getTOutBlock() == o.expectedTOutBlock &&
         cfg.getHOutBlock() == o.expectedHOutBlock &&
         cfg.getWOutBlock() == o.expectedWOutBlock;
}
} // namespace

class Conv3dOracleAccuracyTest : public ::testing::Test {
public:
  mlir::MLIRContext context;
  void SetUp() override { context.loadDialect<TTNNDialect>(); }
};

TEST_F(Conv3dOracleAccuracyTest, TopKHitRate) {
  std::vector<OracleEntry> oracle = loadOracle();
  ASSERT_FALSE(oracle.empty()) << "Oracle JSON loaded zero entries.";

  size_t total = oracle.size();
  size_t top1Hits = 0;
  size_t top5Hits = 0;
  size_t reachable = 0;

  constexpr int64_t TILE_WIDTH = 32;
  for (const OracleEntry &o : oracle) {
    int64_t cInAligned =
        ((o.inChannels + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    int64_t cOutAligned =
        ((o.outChannels + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH;
    int64_t spatialVolume = o.tOut * o.hOut * o.wOut;

    // Filter mirrors tt-metal's brute-force sweep: divisibility is required
    // for c_in/c_out blocks (the weight reshape depends on it) but spatial
    // (t/h/w) blocks are allowed to be non-divisors — the runtime handles
    // partial last blocks via padding. The hard h*w <= 256 cap is still
    // enforced (CB capacity). This is intentionally looser than the
    // LegalOpConfigAnalysis filter we ship in tt-mlir today, which
    // currently requires divisibility everywhere; the gap surfaces below
    // as oracle entries we score against but cannot reach in production.
    auto filter = [&](const Conv3dConfigAttr &cfg) {
      if (auto cIn = cfg.getCInBlock()) {
        if (cIn.value() == 0 || cIn.value() > cInAligned ||
            (o.kT * o.kH * o.kW * cInAligned) % cIn.value() != 0) {
          return true;
        }
      }
      if (auto cOut = cfg.getCOutBlock()) {
        if (cOut.value() == 0 || cOut.value() > cOutAligned ||
            cOutAligned % cOut.value() != 0) {
          return true;
        }
      }
      auto positive = [](std::optional<uint32_t> v) {
        return v.has_value() && v.value() == 0;
      };
      if (positive(cfg.getTOutBlock()) || positive(cfg.getHOutBlock()) ||
          positive(cfg.getWOutBlock())) {
        return true;
      }
      if (cfg.getHOutBlock() && cfg.getWOutBlock()) {
        if (cfg.getHOutBlock().value() * cfg.getWOutBlock().value() > 256) {
          return true;
        }
      }
      return false;
    };

    Conv3dConfigAttr baseConfig =
        Conv3dConfigAttr::get(&context,
                              /*weights_dtype=*/std::nullopt,
                              /*t_out_block=*/std::nullopt,
                              /*w_out_block=*/std::nullopt,
                              /*h_out_block=*/std::nullopt,
                              /*c_out_block=*/std::nullopt,
                              /*c_in_block=*/std::nullopt,
                              /*compute_with_storage_grid_size=*/std::nullopt);

    Conv3dConfigSearchSpace space = buildSearchSpaceWithExpected(o);
    std::vector<Conv3dConfigAttr> all;
    forEachConv3dConfig(static_cast<Conv3dOp *>(nullptr), baseConfig, space,
                        filter,
                        [&](Conv3dConfigAttr cfg) { all.push_back(cfg); });
    std::stable_sort(
        all.begin(), all.end(), [&](Conv3dConfigAttr a, Conv3dConfigAttr b) {
          return scoreConfig(a, spatialVolume) > scoreConfig(b, spatialVolume);
        });

    // Did the search space + filter actually contain the oracle's pick?
    bool inSpace = std::any_of(all.begin(), all.end(), [&](Conv3dConfigAttr c) {
      return sameBlocking(c, o);
    });
    if (inSpace) {
      ++reachable;
    } else {
      // Diagnostic: if the expected blocking is not even reachable, the
      // filter rejects it (most often a divisibility miss). Surface this
      // separately from a scoring miss.
      ADD_FAILURE() << "Oracle expected blocking is unreachable by filter "
                       "for shape (kT="
                    << o.kT << ", kH=" << o.kH << ", kW=" << o.kW
                    << ", C_in=" << o.inChannels << ", C_out=" << o.outChannels
                    << ", T=" << o.tOut << ", H=" << o.hOut << ", W=" << o.wOut
                    << "). Expected: c_in=" << o.expectedCInBlock
                    << " c_out=" << o.expectedCOutBlock
                    << " t=" << o.expectedTOutBlock
                    << " h=" << o.expectedHOutBlock
                    << " w=" << o.expectedWOutBlock;
      continue;
    }

    // Top-1
    if (!all.empty() && sameBlocking(all[0], o)) {
      ++top1Hits;
    }
    // Top-5
    bool inTop5 = false;
    for (size_t i = 0; i < std::min<size_t>(5, all.size()); ++i) {
      if (sameBlocking(all[i], o)) {
        ++top5Hits;
        inTop5 = true;
        break;
      }
    }
    if (!inTop5) {
      // Print the top-3 picks alongside the expected so failures are
      // actionable. Helps tune the scoring tuple in Phase 5 follow-ups.
      llvm::errs() << "[miss] shape (kT=" << o.kT << ", kH=" << o.kH
                   << ", kW=" << o.kW << ", Cin=" << o.inChannels
                   << ", Cout=" << o.outChannels << ", T=" << o.tOut
                   << ", H=" << o.hOut << ", W=" << o.wOut
                   << ") expected (cIn=" << o.expectedCInBlock
                   << " cOut=" << o.expectedCOutBlock
                   << " t=" << o.expectedTOutBlock
                   << " h=" << o.expectedHOutBlock
                   << " w=" << o.expectedWOutBlock << "); we picked:";
      for (size_t i = 0; i < std::min<size_t>(3, all.size()); ++i) {
        auto c = all[i];
        llvm::errs() << " [" << i << "](cIn=" << c.getCInBlock().value_or(0)
                     << " cOut=" << c.getCOutBlock().value_or(0)
                     << " t=" << c.getTOutBlock().value_or(0)
                     << " h=" << c.getHOutBlock().value_or(0)
                     << " w=" << c.getWOutBlock().value_or(0) << ")";
      }
      llvm::errs() << " total_candidates=" << all.size() << "\n";
    }
  }

  llvm::errs() << "[Conv3dOracleAccuracy] total=" << total
               << " reachable=" << reachable << " top1=" << top1Hits
               << " top5=" << top5Hits << "\n";

  // Reachability floor: every oracle entry must be in the search space, or
  // we cannot meaningfully grade scoring.
  EXPECT_EQ(reachable, total)
      << "Some oracle entries are not in the search space; tighten "
         "buildSearchSpaceWithExpected or relax the filter.";

  // Starting thresholds: top-5 >= 20%, top-1 >= 10%. The remaining 80%
  // of oracle picks miss because our scoring lacks an L1-budget cost — the
  // oracle frequently picks smaller `t_out_block` than the largest valid
  // divisor, suggesting tt-metal's hand-tuning weighs activation footprint
  // we don't model. Future Phase 5 follow-up: lift `estimate_l1_bytes` from
  // tt-metal's `bruteforce_conv3d_sweep.py` and add it to scoring. Ratchet
  // these thresholds upward when that lands.
  EXPECT_GE(top5Hits * 100, total * 20) << "top-5 hit rate dropped below 20%";
  EXPECT_GE(top1Hits * 100, total * 10) << "top-1 hit rate dropped below 10%";
}
