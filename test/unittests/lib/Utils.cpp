// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "testing/Utils.h"

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mlir::tt::testing {

#define TT_ENV_TEST_SEED "TTMLIR_TEST_SEED"
#define TT_ENV_TEST_WORKFLOW "TTMLIR_TEST_WORKFLOW"
#define TT_TEST_WORKFLOW_NIGHTLY "nightly"
#define TT_TEST_SEED_KEY "seed"

static std::uint64_t getRandomSeed() {
  std::uint64_t seed = 0;

  const char *seedOverride = std::getenv(TT_ENV_TEST_SEED);
  // If seed is overridden by the env, assume we're running in repro mode
  // and parse the override string as-is.
  if (seedOverride != nullptr && seedOverride[0]) {
    std::from_chars(seedOverride, seedOverride + std::strlen(seedOverride),
                    seed);
  }
  if (!seed) {
    // If no env override or its parsing failed, generate a seed value.
    const char *workflow = std::getenv(TT_ENV_TEST_WORKFLOW);
    // (a) If running in nightly workflow, generate a time-varying value.
    if (workflow != nullptr &&
        !std::strncmp(workflow, TT_TEST_WORKFLOW_NIGHTLY,
                      sizeof(TT_TEST_WORKFLOW_NIGHTLY))) {
      // Use hr clock, make sure some low and some high bits are always set.
      seed = (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count() |
              (1L << 53) | (1 << 13));

      // Scramble `seed` some more, this is scheme M8+A2(a=8,b=31,c=17)
      // from Table V of "An experimental exploration of Marsaglia’s xorshift
      // generators, scrambled" (https://arxiv.org/pdf/1402.6246).
      constexpr std::uint64_t M8 = 1181783497276652981;
      for (int32_t i = 0; i < 100; ++i) {
        seed ^= seed << 17; // c
        seed ^= seed >> 31; // b
        seed ^= seed << 8;  // a
        seed *= M8;
      }
    } else {
      // (b) Otherwise, use a hardcoded const.
      seed = 16986427154618410377UL;
    }
  }

  return seed;
}

#undef TT_TEST_WORKFLOW_NIGHTLY
#undef TT_ENV_TEST_WORKFLOW
#undef TT_ENV_TEST_SEED

// A global singleton object to help obtain test env settings and log some of
// them at both the beginning and the end of normal gtest stdout output.
namespace {
struct TestEnv {
  TestEnv() : seed(getRandomSeed()) {
    std::cout << "[testsuite random seed: " << seed << "]\n";
  }

  ~TestEnv() { std::cout << "[testsuite random seed: " << seed << "]\n"; }

  std::uint64_t seed;
};
} // namespace

static const TestEnv &env() {
  static TestEnv instance;
  return instance;
}

std::uint64_t randomSeed() {
  const auto seed = env().seed;
  // Use gtest test property facilities to record test case/test suite random
  // seeds so that they are captured in XML/JSON reports along with standard
  // test results.
  ::testing::Test::RecordProperty(TT_TEST_SEED_KEY, std::to_string(seed));
  return seed;
}

#undef TT_TEST_SEED_KEY

} // namespace mlir::tt::testing
