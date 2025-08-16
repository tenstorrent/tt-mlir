// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/AllocationPlanner.h"

#include "ttmlir/Dialect/TTIR/Analysis/AllocationDefs.h"
#include "ttmlir/Dialect/TTIR/Analysis/AllocationTools.h"

#include "testing/Utils.h"

#include <cmath>
#include <cstdint>

namespace mlir::tt::ttir {

namespace gtest = ::testing;

#define TT_TEST_DEBUG(/* fmt, args */...)                                      \
  TTMLIR_DEBUG(ttmlir::LogComponent::Test, __VA_ARGS__)

using std::int32_t;
using std::int64_t;

using Planner = AllocationPlanner;

//===----------------------------------------------------------------------===//
// Tests for base (single space) algorithms.
//===----------------------------------------------------------------------===//

template <Planner::Algorithm Algorithm>
struct AllocationTestScenario {
  static constexpr Planner::Algorithm algorithm = Algorithm;
};

using AllocationTestScenarios =
    gtest::Types<AllocationTestScenario<Planner::Algorithm::Simple>,
                 AllocationTestScenario<Planner::Algorithm::Greedy>>;

template <typename T>
struct AllocationTest : public gtest::Test {};

TYPED_TEST_SUITE(AllocationTest, AllocationTestScenarios,
                 /* clang variadic macro issue workaround */);

// Test post-construction invariants and some trivial edge cases.
TYPED_TEST(AllocationTest, EdgeCases) {

  using Scenario = TypeParam;

  // An empty allocation problem is valid input.
  {
    Planner::Problem problem;

    ASSERT_TRUE(problem.variables.empty());
    ASSERT_TRUE(problem.bound.empty());

    const Planner::AllocateStats allocateStats =
        Planner::allocate(problem, Scenario::algorithm);
    const Planner::AllocateStats verifyStats = Planner::verify(problem);

    ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
    ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

    ASSERT_EQ(allocateStats.memUsage, 0);
    ASSERT_EQ(allocateStats.maxSize, 0);
    ASSERT_EQ(allocateStats.maxLoad, 0);
  }
  // A problem with variables that have empty scratch domains is equivalent to
  // an empty problem.
  {
    Planner::Problem problem;

    constexpr int32_t varCount = 3;

    for (int32_t i = 0; i < varCount; ++i) {
      problem.def([&](Planner::VariableBuilder &b) {
        b.request(Planner::Space::Spill, 100'000, 1 + i, 2 + i);
        b.place(Planner::Space::Scratch);
      });
    }
    ASSERT_EQ(problem.variables.size(), varCount);
    ASSERT_TRUE(problem.bound.empty());

    const Planner::AllocateStats allocateStats =
        Planner::allocate(problem, Scenario::algorithm);
    const Planner::AllocateStats verifyStats = Planner::verify(problem);

    ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
    ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

    ASSERT_EQ(allocateStats.memUsage, 0);
    ASSERT_EQ(allocateStats.maxSize, 0);
    ASSERT_EQ(allocateStats.maxLoad, 0);
  }
  // A problem with variables that have non-empty scratch domains but are
  // all bound to the spill space is equivalent to en empty problem.
  {
    Planner::Problem problem;

    constexpr int32_t varCount = 3;

    for (int32_t i = 0; i < varCount; ++i) {
      problem.def([&](Planner::VariableBuilder &b) {
        b.request(Planner::Space::Scratch, 100'000, 1 + i, 2 + i);
        b.bind(Planner::Space::Spill);
      });
    }
    ASSERT_EQ(problem.variables.size(), varCount);
    ASSERT_EQ(problem.bound.size(), varCount);

    const Planner::AllocateStats allocateStats =
        Planner::allocate(problem, Scenario::algorithm);
    const Planner::AllocateStats verifyStats = Planner::verify(problem);

    ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
    ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

    ASSERT_EQ(allocateStats.memUsage, 0);
    ASSERT_EQ(allocateStats.maxSize, 0);
    ASSERT_EQ(allocateStats.maxLoad, 0);
  }
  // A one-allocation problem has a trivial outcome regardless of the algorithm.
  {
    Planner::Problem problem;

    // Add a single (unbound) variable with a single request in scratch space.

    constexpr Planner::AllocSizeT scratchRequest = 100'000;

    problem.def([](Planner::VariableBuilder &b) {
      b.request(Planner::Space::Scratch, scratchRequest, 1, 2);
      b.request(Planner::Space::Spill, 10 * scratchRequest, 1, 2);
      b.place(Planner::Space::Scratch);
    });

    ASSERT_EQ(problem.variables.size(), 1);

    const Planner::AllocateStats allocateStats =
        Planner::allocate(problem, Scenario::algorithm);
    const Planner::AllocateStats verifyStats = Planner::verify(problem);

    ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
    ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

    ASSERT_EQ(problem.requests[0].offset, 0)
        << "single request should have been allocated at zero offset";
    ASSERT_EQ(allocateStats.memUsage, scratchRequest);
    ASSERT_EQ(allocateStats.maxSize, scratchRequest);
  }
}

// Run all algorithms against request sequences with varying density over scope
// time axis and varying probability of liveness conflicts.
TYPED_TEST(AllocationTest, Conflicts) {

  using Scenario = TypeParam;

  constexpr int32_t repeats = 3;

  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);

  constexpr Planner::AllocSizeT sizeLimit = 1000;
  constexpr Planner::SequenceT positionLimit = 1000; // Soft limit.

  std::uniform_int_distribution<int32_t> unifSize(1, sizeLimit -
                                                         1); // [1, sizeLimit)
  std::uniform_int_distribution<int32_t> unifPosition(
      0, positionLimit); // [1, positionLimit]
  std::uniform_real_distribution<> unif(0.0, 1.0);

  // Loosely control the expected probability of conflicts by sprinkling request
  // intervals using a random walk of sorts, with varying "drift" of the
  // interval midpoint.

  for (int32_t n : {10, 100, 1000}) {
    for (int32_t drift : {2, 10, 100}) {
      for (int32_t durationScale : {drift / 2, drift, drift * 2}) {
        for (int32_t repeat = 0; repeat < repeats; ++repeat) {
          TT_TEST_DEBUG("[{}/{}] sub-scenario: n = {}, drift = {}, "
                        "durationScale = {} ...",
                        (repeat + 1), repeats, n, drift, durationScale);
          gen.seed(seed++);

          Planner::Problem problem;
          {
            Planner::SequenceT midPrev = unifPosition(gen);

            for (auto i = 0; i < n; ++i) {
              const Planner::AllocSizeT size = unifSize(gen);

              const Planner::SequenceT mid =
                  std::abs(static_cast<Planner::SequenceT>(
                      midPrev + drift * (0.5 - unif(gen)))) %
                  positionLimit;

              const Planner::SequenceT halfDuration = durationScale * unif(gen);
              const Planner::SequenceT first = std::max(0, mid - halfDuration);
              const Planner::SequenceT last =
                  std::min(positionLimit, mid + 1 + halfDuration);

              problem
                  .variable(problem.def([&](Planner::VariableBuilder &builder) {
                    builder.request(Planner::Space::Scratch, size, first, last);
                  }))
                  .placement = Planner::Space::Scratch;

              midPrev = mid;
            }
          }
          ASSERT_EQ(problem.variables.size(), n);

          const Planner::AllocateStats allocateStats =
              Planner::allocate(problem, Scenario::algorithm);
          const Planner::AllocateStats verifyStats = Planner::verify(problem);

          // Verification will compute the same max size and mem usage.

          ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
          ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

          ASSERT_GE(verifyStats.usageRatio(), 1.0)
              << "inconsistent max load/mem usage calculation";
        }
      }
    }
  }
}

// When no allocation requests have actual live conflicts, any non-naive
// algorithm should produce a plan that
//  - has max memory usage equal to the largest request size;
//  - verifies to a max mem/max load ratio of exactly 1.0.
// This test verifies these expectations for the default (Greedy) algorithm.
TEST(GreedyAllocationTest, ConflictFree) {

  constexpr auto algorithm = Planner::Algorithm::Greedy;

  constexpr int32_t repeats = 3;

  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);

  constexpr Planner::AllocSizeT sizeLimit = 1000;
  constexpr Planner::SequenceT positionLimit = 1000; // Soft limit.

  std::uniform_int_distribution<int32_t> unifSize(1, sizeLimit -
                                                         1); // [1, sizeLimit)

  for (int32_t repeat = 0; repeat < repeats; ++repeat) {
    gen.seed(seed + repeat);
    std::uniform_int_distribution<int32_t> unifLength(0, 5 + 2 * repeat);

    Planner::AllocSizeT maxSize = 0;
    Planner::Problem problem;
    {
      Planner::SequenceT first = 0;
      while (first < positionLimit) {
        const Planner::AllocSizeT size = unifSize(gen);
        const Planner::SequenceT last = first + unifLength(gen);

        problem
            .variable(problem.def([&](Planner::VariableBuilder &builder) {
              builder.request(Planner::Space::Scratch, size, first, last);
            }))
            .placement = Planner::Space::Scratch;

        maxSize = std::max(maxSize, size);
        first = last + 1 + unifLength(gen); // Ensure no position overlap.
      }
    }

    const Planner::AllocateStats allocateStats =
        Planner::allocate(problem, algorithm);

    ASSERT_EQ(allocateStats.memUsage, maxSize);
    ASSERT_EQ(allocateStats.maxSize, maxSize);

    const Planner::AllocateStats verifyStats = Planner::verify(problem);

    ASSERT_EQ(verifyStats.usageRatio(), 1.0)
        << "expected max load/mem usage ratio of exactly 1.0";
  }
}

//===----------------------------------------------------------------------===//
// Tests for space mapping algorithms.
//===----------------------------------------------------------------------===//

TEST(AllocationToolsTest, SupportTypes) {

  auto seed = tt::testing::randomSeed();

  for (double bindFraction : {0.0, 0.25, 1.0}) {
    Planner::Problem problem = AllocationTools::generate(
        {{{1, 5}, {15, 19}, {3, 5}}, bindFraction, seed});

    Planner::Problem copy;
    copy = problem;
    ASSERT_EQ(problem, copy);

    if (!problem.bound.empty()) {
      copy.bound.clear();
      ASSERT_FALSE(problem == copy);
    }
  }
}

// Serialize 'Planner::Problem' as JSON and parse it back.
TEST(AllocationToolsTest, JSONSerialization) {

  auto seed = tt::testing::randomSeed();

  for (double bindFraction : {0.0, 0.5, 1.0}) {
    Planner::Problem problem = AllocationTools::generate(
        {{{1, 5}, {15, 19}, {3, 5}}, bindFraction, seed});

    std::stringstream s;
    AllocationTools::write(problem, s);

    auto parsed = AllocationTools::read(s);
    if (auto e = parsed.takeError()) {
      GTEST_FAIL() << "failed to parse JSON: " << e;
    }
    ASSERT_EQ(problem, *parsed);
  }
}

namespace {

// Calculate mem usage feasibility range for `problem`.
std::array<Planner::AllocSizeT, 2> boundMemUsage(Planner::Problem &problem,
                                                 Planner::Algorithm algorithm) {
  std::array<Planner::AllocSizeT, 2> r{-1, -1};

  problem.reset(Planner::Space::Spill);
  r[0] = Planner::allocate(problem, algorithm).memUsage;

  problem.reset(Planner::Space::Scratch);
  r[1] = Planner::allocate(problem, algorithm).memUsage;

  return r;
}

// Divide the mem usage feasibility interval for `problem` into `limitCount`
// values and drive `Planner::solve()` through each.
void runSolveTest(Planner::Problem &problem, int32_t limitCount) {
  TT_TEST_DEBUG("using problem config with {} variable(s), {} request(s)",
                problem.variables.size(), problem.requests.size());

  constexpr Planner::Algorithm algorithm = Planner::Algorithm::Greedy;

  // Remember the set of bound variables to check later that they were not
  // rebound by solve().

  llvm::DenseMap<Planner::IndexT, Planner::Space> bindings;
  for (const auto varIndex : problem.bound) {
    bindings[varIndex] = problem.variables[varIndex].placement;
  }

  // Since generate() uses randomized request sizes, to keep testing meaningful
  // memory pressures start by measuring feasible range of mem usage.

  std::array<Planner::AllocSizeT, 2> memUsageBounds =
      boundMemUsage(problem, algorithm);

  TT_TEST_DEBUG("mem usage bounded as [{}, {}]", memUsageBounds[0],
                memUsageBounds[1]);

  TT_assert(memUsageBounds[0] > 0);
  TT_assert(memUsageBounds[0] <= memUsageBounds[1]);

  // Now solve() the problem using a series of progressively tigher mem usage
  // limits. (Note that both lower and upper feasible limits are used as
  // inclusive bounds, with some additional test validation.)

  std::vector<Planner::AllocSizeT> limits;
  if (memUsageBounds[1] - memUsageBounds[0] > 100) {
    limits.resize(limitCount);
    const double step =
        std::log(static_cast<double>(memUsageBounds[1]) / memUsageBounds[0]) /
        (limitCount - 1);

    limits[0] = memUsageBounds[1];
    limits[limitCount - 1] = memUsageBounds[0];

    for (int32_t l = 1; l < limitCount - 1; ++l) {
      limits[l] = std::round(limits[0] * std::exp(-(l * step)));
    }
  } else {
    limitCount = 1;
    limits.push_back(memUsageBounds[1]);
  }

  for (int32_t l = 0; l < limitCount; ++l) {
    problem.reset();

    const Planner::SolveStats solveStats =
        Planner::spillAllocate(problem, limits[l]);
    TT_TEST_DEBUG("[limit {}] took {} step(s), spilled {} variable(s)",
                  limits[l], solveStats.stepsTaken, solveStats.spillCount);

    ASSERT_LE(solveStats.memUsage, limits[l])
        << "should have converged to a feasible solution";

    llvm::DenseMap<Planner::IndexT, Planner::Space> bindingsAfter;
    for (const auto varIndex : problem.bound) {
      bindingsAfter[varIndex] = problem.variables[varIndex].placement;
    }
    ASSERT_EQ(bindingsAfter, bindings)
        << "should not have changed variable bindings";

    if (l == 0) {
      ASSERT_EQ(solveStats.spillCount, 0)
          << "should not have spilled any variables at the upper mem usage "
             "limit";
    }

    const Planner::AllocateStats verifyStats = Planner::verify(problem);
    ASSERT_EQ(solveStats.memUsage, verifyStats.memUsage);
  }
}

} // namespace

// Generate a number of mem planning problems of varying conflict geometry
// and solve each over its range of mem usage limits.
TEST(AllocationMappingTest, RandomizedProblems) {

#ifdef TT_TESTING_DEBUG
  constexpr int32_t repeats = 20;
#else
  constexpr int32_t repeats = 100;
#endif

  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);

  auto unif = [&gen](int32_t a, int32_t b) {
    return std::uniform_int_distribution<int32_t>(a, b)(gen);
  };

  for (int32_t repeat = 0; repeat < repeats; ++repeat) {
    // Make a random problem config.
    for (double bindFraction : {0.0, 0.5, 1.0}) {
      const int32_t segmentCount = unif(1, 10);
      llvm::SmallVector<AllocationTools::GenerateSegmentParms> segmentCfgs;

      for (int32_t segment = 0; segment < segmentCount; ++segment) {
        const int32_t neckLength = unif(1, 5);
        const int32_t conflictCount = unif(1, 30);
        segmentCfgs.emplace_back(
            AllocationTools::GenerateSegmentParms{neckLength, conflictCount});
      }

      const AllocationTools::GenerateCfg cfg(std::move(segmentCfgs),
                                             bindFraction, seed + repeat);
      TT_TEST_DEBUG("scenario {} cfg: {{{}}}", repeat, cfg);

      Planner::Problem problem = AllocationTools::generate(cfg);
      runSolveTest(problem, 5);
    }
  }
}

// Run some large-ish problems, with O(10^4) requests.
TEST(AllocationMappingTest, LargeProblems) {

#ifdef TT_TESTING_DEBUG
  constexpr int32_t targetVarCount = 500;
#else
  constexpr int32_t targetVarCount = 5000;
#endif

  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);

  auto unif = [&gen](int32_t a, int32_t b) {
    return std::uniform_int_distribution<int32_t>(a, b)(gen);
  };

  constexpr double bindFraction = 0.25;

  for (int32_t segmentCount = 5; segmentCount >= 1; segmentCount -= 2) {
    const int32_t segmentVarCount = targetVarCount / segmentCount;

    llvm::SmallVector<AllocationTools::GenerateSegmentParms> segmentCfgs;
    for (int32_t segment = 0; segment < segmentCount; ++segment) {
      TT_assert(segmentVarCount >= 3);
      const int32_t neckLength = unif(1, segmentVarCount / 3);
      const int32_t conflictCount = (segmentVarCount - neckLength) / 2;
      segmentCfgs.emplace_back(
          AllocationTools::GenerateSegmentParms{neckLength, conflictCount});
    }

    const AllocationTools::GenerateCfg cfg{std::move(segmentCfgs), bindFraction,
                                           seed + segmentCount};
    TT_TEST_DEBUG("scenario segment count {}, cfg: {{{}}}", segmentCount, cfg);

    Planner::Problem problem = AllocationTools::generate(cfg);
    runSolveTest(problem, 4);
  }
}

} // namespace mlir::tt::ttir
