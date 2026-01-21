// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_TESTING_UTILS_H
#define TT_TESTING_UTILS_H

#include "ttmlir/Asserts.h"

// A convenience include so that tests only need "testing/Utils.h".
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace mlir::tt::testing {

// "TT_TESTING_DEBUG" is meant to be defined in build variants compiled
// without optimization. This is useful for scaling test parameters such
// that randomized sampling/coverage can be kept high in release builds
// while keeping unit testing snappy in dev workcycles.
#ifndef __OPTIMIZE__
#define TT_TESTING_DEBUG
#endif

// Helper macro for logging on behalf of `ttmlir::LogComponent::Test`.
#define TT_TEST_DEBUG(/* fmt, args */...)                                      \
  TTMLIR_DEBUG(ttmlir::LogComponent::Test, __VA_ARGS__)

// Return a 64-bit random seed that can be used by randomized tests. This value
// is a sigleton in the current process and it generated according to the
// following rules:
//
// (a) by default, this is a constant that is hardcoded/doesn't vary with time;
//
// (b) if the  process is running in nightly mode (indicated by env var
// "TTMLIR_TEST_WORKFLOW" set to "nightly"), the value will vary for each
// process run;
//
// (c) if the process is running in repro mode (indicated by env var
// "TTMLIR_TEST_SEED" set to a 64-bit int string), the value will be set to this
// override.
//
// This function also ensures that the seed is captured in gtest reports as test
// properties at the scope (test/suite/etc) from which it is invoked. These will
// show up in the JUnit XML report as
// clang-format off
//  <properties>
//    <property name="seed" value="1756947448940470413"/>
//  </properties>
// clang-format on
// When gtest is driven by llvm-lit, the latter tool builds its own
// JUnit-compatible reports which do not propagate generic key/value properties
// of test/suites. In such setups, however, the seed should be recoverable from
// the stdout capture of test failure messages.
std::uint64_t randomSeed();

//===----------------------------------------------------------------------===//
namespace impl {

using std::uint32_t;
using std::uint64_t;

/// Uniform random bit generators (URBGs) based on "xorshift" random generators
/// that are drop-in compatible with std <random>.
///
/// The motivation for using these as defaults for randomized testing is that
/// these generators are guaranteed to have reproducible implementations
/// regardless of the compiler/stdlib version and have proven good statistical
/// properties with low latency and minimal state.
///
/// @see [Marsaglia 2003] "Xorshift RNGs"
/// (https://www.jstatsoft.org/article/view/v008i14/916)
/// @see [Vigna 2016] "An experimental exploration of Marsaglia’s xorshift
/// generators, scrambled"  (https://arxiv.org/pdf/1402.6246)

template <typename T>
struct URBGBase {
  static_assert(std::is_integral_v<T>);

  using result_type = T;

  static constexpr result_type min() {
    return std::numeric_limits<result_type>::min();
  }
  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }
};

template <typename T>
struct URBGXorshift32 final : URBGBase<T> {

  static_assert(sizeof(T) == 4);
  using Base = URBGBase<T>;

  using Base::max;
  using Base::min;
  using typename Base::result_type;

  URBGXorshift32(T seed) : state(seed) { TT_assert(state != 0u); }

  void seed(result_type seed) {
    TT_assert(seed != 0u);
    state = seed;
  }

  result_type operator()() {
    uint32_t y = state;

    // Taken from p.2 of [Marsaglia 2003].
    y ^= y << 17; // a
    y ^= y >> 15; // b
    y ^= y << 26; // c

    state = y;
    return y;
  }

  uint32_t state;
};

template <typename T>
struct URBGXorshift64 final : URBGBase<T> {

  static_assert(sizeof(T) == 8);
  using Base = URBGBase<T>;

  using Base::max;
  using Base::min;
  using typename Base::result_type;

  URBGXorshift64(T seed) : state(seed) { TT_assert(state != 0u); }

  void seed(result_type seed) {
    TT_assert(seed != 0u);
    state = seed;
  }

  result_type operator()() {
    uint64_t y = state;

    // This is M32 A1(12, 25, 27) from Table V/Fig. 10 of [Vigna 2016].
    y ^= y >> 12; // a
    y ^= y << 25; // b
    y ^= y >> 27; // c

    state = y;
    return static_cast<T>(y * M32);
  }

  static constexpr uint64_t M32 = 2685821657736338717;

  uint64_t state;
};
//===----------------------------------------------------------------------===//

template <typename ResultType, typename = void>
struct select_default_RNG {};

template <typename ResultType>
struct select_default_RNG<ResultType,
                          std::enable_if_t<std::is_integral_v<ResultType> &&
                                           (sizeof(ResultType) == 4)>> {
  using type = URBGXorshift32<ResultType>;
};

template <typename ResultType>
struct select_default_RNG<ResultType,
                          std::enable_if_t<std::is_integral_v<ResultType> &&
                                           (sizeof(ResultType) == 8)>> {
  using type = URBGXorshift64<ResultType>;
};

template <typename ResultType>
using select_default_RNG_t =
    typename impl::select_default_RNG<ResultType>::type;

} // namespace impl
//===----------------------------------------------------------------------===//

// Return a new instance of a random number generator
// suitable for use with <random> facilities.
template <typename ResultType>
auto createRNG(ResultType seed) -> impl::select_default_RNG_t<ResultType> {
  return impl::select_default_RNG_t<ResultType>(seed);
}

} // namespace mlir::tt::testing

#endif // TT_TESTING_UTILS_H
