// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_TESTING_UTILS_H
#define TT_TESTING_UTILS_H

// A convenience include so that tests only need "testing/Utils.h".
#include "llvm-gtest/gtest/gtest.h"

#include <cstdint>

namespace mlir::tt::testing {

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
// show up in the XML report as
// clang-format off
//  <properties>
//    <property name="seed" value="1756947448940470413"/>
//  </properties>
// clang-format on
extern std::uint64_t randomSeed();

} // namespace mlir::tt::testing

#endif // TT_TESTING_UTILS_H
