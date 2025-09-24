// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_TESTING_UTILS_H
#define TT_TESTING_UTILS_H

// A convenience include so that tests only need "testing/Utils.h".
#include "gtest/gtest.h"

#include <cstdint>
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

namespace impl {

// TODO(#3531) replace this with xorshift impl
template <typename ResultType, typename = void>
struct select_default_RNG {};

template <typename ResultType>
struct select_default_RNG<ResultType,
                          std::enable_if_t<std::is_integral_v<ResultType> &&
                                           (sizeof(ResultType) == 4)>> {
  using type = std::mt19937;
};

template <typename ResultType>
struct select_default_RNG<ResultType,
                          std::enable_if_t<std::is_integral_v<ResultType> &&
                                           (sizeof(ResultType) == 8)>> {
  using type = std::mt19937_64;
};

template <typename ResultType>
using select_default_RNG_t =
    typename impl::select_default_RNG<ResultType>::type;

} // namespace impl

// Return a new instance of a random number generator
// suitable for use with <random> facilities.
template <typename ResultType>
auto createRNG(ResultType seed) -> impl::select_default_RNG_t<ResultType> {
  return impl::select_default_RNG_t<ResultType>(seed);
}

} // namespace mlir::tt::testing

#endif // TT_TESTING_UTILS_H
