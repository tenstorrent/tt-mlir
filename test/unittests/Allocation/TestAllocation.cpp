// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/AllocationPlanner.h"

#include "ttmlir/Support/Logger.h"

#include "testing/Utils.h"

#include <cstdint>
#include <random>

namespace mlir::tt::ttir {

namespace gtest = ::testing;

template <AllocationPlanner::Algorithm Algorithm>
struct TestScenario {
  static constexpr AllocationPlanner::Algorithm algorithm = Algorithm;
};

using TestScenarios =
    gtest::Types<TestScenario<AllocationPlanner::Algorithm::Simple>,
                 TestScenario<AllocationPlanner::Algorithm::Greedy>>;

template <typename T>
struct AllocationTest : public gtest::Test {};
TYPED_TEST_SUITE(AllocationTest, TestScenarios,
                 /* clang variadic macro issue workaround */);

TYPED_TEST(AllocationTest, EdgeCases) {

  using Scenario = TypeParam;

  // An empty allocation plan is valid input.
  {
    AllocationPlanner::Context ctx;
    ASSERT_EQ(ctx.size(), 0);

    const AllocationPlanner::Stats allocateStats =
        AllocationPlanner::allocate(ctx, Scenario::algorithm);
    const AllocationPlanner::Stats verifyStats = AllocationPlanner::verify(ctx);

    ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
    ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

    ASSERT_EQ(allocateStats.memUsage, 0);
    ASSERT_EQ(allocateStats.maxSize, 0);
    ASSERT_EQ(allocateStats.maxLoad, 0);
  }
  // A one-allocation plan has a trivial outcome regardless of the algorithm.
  {
    AllocationPlanner::Context ctx;

    ctx.add(100'000, 1, 2);
    ASSERT_EQ(ctx.size(), 1);

    const AllocationPlanner::Stats allocateStats =
        AllocationPlanner::allocate(ctx, Scenario::algorithm);
    const AllocationPlanner::Stats verifyStats = AllocationPlanner::verify(ctx);

    ASSERT_EQ(allocateStats.maxSize, verifyStats.maxSize);
    ASSERT_EQ(allocateStats.memUsage, verifyStats.memUsage);

    ASSERT_EQ(ctx[0].offset, 0)
        << "single request should have been allocated at zero offset";
  }
}

// Run all algorithms against request sequences with varying density over scope
// time axis and varying probability of liveness conflicts.
TYPED_TEST(AllocationTest, Conflicts) {

  using Scenario = TypeParam;

  constexpr std::int32_t repeats = 3;
  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);

  constexpr AllocationPlanner::AllocSizeT sizeLimit = 1000;
  constexpr AllocationPlanner::SequenceT positionLimit = 1000; // Soft limit.

  std::uniform_int_distribution<std::int32_t> unifSize(
      1, sizeLimit - 1); // [1, sizeLimit)
  std::uniform_int_distribution<std::int32_t> unifPosition(
      0, positionLimit); // [1, positionLimit]
  std::uniform_real_distribution<> unif(0.0, 1.0);

  // Loosely control the expected probability of conflicts by sprinkling request
  // intervals using a random walk of sorts, with varying "drift" of the
  // interval midpoint.

  for (std::int32_t n : {10, 100, 1000}) {
    for (std::int32_t drift : {2, 10, 100}) {
      for (std::int32_t durationScale : {drift / 2, drift, drift * 2}) {
        for (std::int32_t repeat = 0; repeat < repeats; ++repeat) {
          TTMLIR_DEBUG(ttmlir::LogComponent::Test,
                       "[{}/{}] sub-scenario: n = {}, drift = {}, "
                       "durationScale = {} ...",
                       (repeat + 1), repeats, n, drift, durationScale);
          gen.seed(seed++);

          AllocationPlanner::Context ctx;
          {
            AllocationPlanner::SequenceT midPrev = unifPosition(gen);

            for (auto i = 0; i < n; ++i) {
              const AllocationPlanner::AllocSizeT size = unifSize(gen);

              const AllocationPlanner::SequenceT mid =
                  std::abs(static_cast<AllocationPlanner::SequenceT>(
                      midPrev + drift * (0.5 - unif(gen)))) %
                  positionLimit;

              const AllocationPlanner::SequenceT halfDuration =
                  durationScale * unif(gen);
              const AllocationPlanner::SequenceT first =
                  std::max(0, mid - halfDuration);
              const AllocationPlanner::SequenceT last =
                  std::min(positionLimit, mid + 1 + halfDuration);

              ctx.add(size, first, last);

              midPrev = mid;
            }
          }
          ASSERT_EQ(ctx.size(), n);

          const AllocationPlanner::Stats allocateStats =
              AllocationPlanner::allocate(ctx, Scenario::algorithm);
          const AllocationPlanner::Stats verifyStats =
              AllocationPlanner::verify(ctx);

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
//  - verifies to max mem/max load ratio of exactly 1.0.
// This test verifies these expectations for the default algorithm.
TEST(GreedyAllocationTest, ConflictFree) {

  constexpr std::int32_t repeats = 3;
  auto seed = tt::testing::randomSeed();
  auto gen = tt::testing::createRNG(seed);

  constexpr AllocationPlanner::AllocSizeT sizeLimit = 1000;
  constexpr AllocationPlanner::SequenceT positionLimit = 1000; // Soft limit.

  std::uniform_int_distribution<std::int32_t> unifSize(
      1, sizeLimit - 1); // [1, sizeLimit)

  for (std::int32_t repeat = 0; repeat < repeats; ++repeat) {
    gen.seed(seed + repeat);
    std::uniform_int_distribution<std::int32_t> unifLength(0, 5 + 2 * repeat);

    AllocationPlanner::AllocSizeT maxSize = 0;
    AllocationPlanner::Context ctx;
    {
      AllocationPlanner::SequenceT first = 0;
      while (first < positionLimit) {
        const AllocationPlanner::AllocSizeT size = unifSize(gen);
        const AllocationPlanner::SequenceT last = first + unifLength(gen);

        ctx.add(size, first, last);

        maxSize = std::max(maxSize, size);
        first = last + 1 + unifLength(gen); // Ensure no position overlap.
      }
    }

    const AllocationPlanner::Stats allocateStats =
        AllocationPlanner::allocate(ctx);

    ASSERT_EQ(allocateStats.memUsage, maxSize);
    ASSERT_EQ(allocateStats.maxSize, maxSize);

    const AllocationPlanner::Stats verifyStats = AllocationPlanner::verify(ctx);

    ASSERT_EQ(verifyStats.usageRatio(), 1.0)
        << "expected max load/mem usage ratio of exactly 1.0";
  }
}

} // namespace mlir::tt::ttir
