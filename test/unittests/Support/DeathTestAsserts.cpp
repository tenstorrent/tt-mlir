// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"

#include "testing/Utils.h"

#include <string>
#include <string_view>

#if !defined(TT_ASSERT_DISABLE_ASSERTS)

// A subset of "TestAsserts.cpp" test suite to verify that assert
// failure messages are seen in stderr when the process terminates.

#define TT_STANDARD_ASSERT_MSG_PREFIX(condition)                               \
  "Assertion `" #condition "` failed"                                          \
  ".*"

TEST(AssertsDeathTest, BinaryExprDecomposition) {
  ASSERT_DEATH(
      {
        int32_t x = 1;
        int32_t y = 2;
        TT_assert(x == y);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(x == y) "1 == 2");
  ASSERT_DEATH(
      {
        int32_t x = 1;
        int32_t y = 2;
        TT_assert(x > y);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(x > y) "1 > 2");
  ASSERT_DEATH(
      {
        int32_t x = 1;
        int32_t y = 2;
        TT_assert(y <= x);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(y <= x) "2 <= 1");
  // With custom formatted context.
  ASSERT_DEATH(
      {
        int32_t x = 1;
        int32_t y = 2;
        TT_assertv(x == y, "x-y was {}", x - y);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(x == y) "x-y was -1");
  // Strings and views.
  ASSERT_DEATH(
      {
        std::string s = "hello";
        TT_assert(s != "hello");
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(s != "hello") "hello != hello");
  ASSERT_DEATH(
      {
        using namespace std::literals;

        auto sa = "abcdef"sv;
        const std::string sb = "asdf";
        TT_assert(sa == sb);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(sa == sb) "abcdef == asdf");
  // Literals.
  ASSERT_DEATH(
      {
        // Note that the stringifier macro will capture the source
        // test but the runtime message will show the evaluated
        // literal. This is ok.
        TT_assert(2 + 2 == 3);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(2 \\+ 2 == 3) "4 == 3");
  // Chars.
  ASSERT_DEATH(
      {
        int8_t a = 'A';
        char b = 'B';
        int8_t c = b;
        TT_assertv(a == b,
                   "hmm, I think a, b, and c were {}, {}, and {}, respectively",
                   a, b, c);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(
          a == b) "hmm, I think a, b, and c were A, B, and B, respectively");
  // llvm streams have issues printing std::nullptr_t, but TT_assert*()s don't.
  ASSERT_DEATH(
      {
        int32_t x = 42;
        void *y = &x;
        TT_assert(y == nullptr);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(y == nullptr) "0x.+ == nullptr");
}

TEST(AssertsDeathTest, IntegralRangeChecks) {
  ASSERT_DEATH(
      {
        int32_t y = 20;
        TT_assert_limit(y, 20);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(
          in_open_range\\(y, 0, 20\\)) "y \\(20\\) is not in \\[0, 20\\)");
  ASSERT_DEATH(
      {
        int32_t y = 2;
        TT_assert_open_range(y, 0, 2);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(
          in_open_range\\(y, 0, 2\\)) "y \\(2\\) is not in \\[0, 2\\)");
  ASSERT_DEATH(
      {
        int32_t y = 2;
        TT_assert_inclusive_range(y, 0, 1);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(
          in_inclusive_range\\(y, 0, 1\\)) "y \\(2\\) is not in \\[0, 1\\]");
  ASSERT_DEATH(
      {
        int32_t y = 2;
        TT_assert_exclusive_range(y, 2, 3);
      },
      TT_STANDARD_ASSERT_MSG_PREFIX(
          in_exclusive_range\\(y, 2, 3\\)) "y \\(2\\) is not in \\(2, 3\\)");
}

#undef TT_STANDARD_ASSERT_MSG_PREFIX

#endif // TT_ASSERT_DISABLE_ASSERTS

// Check that TT-debug* asserts are appropriately elided according
// to TT_ASSERT_ENABLE_DEBUG_ASSERTS.
//
// See also TestAssertsElision.cpp.

TEST(AssertsDeathTest, MacroElisionDebug) {
#if !defined(TT_ASSERT_ENABLE_DEBUG_ASSERTS)
  // TT_debug* should be elided and not abort the process,
  // regardless of TT_ASSERT_DISABLE_ASSERTS status.

  TT_debug(2 < 1);
  TT_debug_limit(10, 1);
  TT_debugv(2 + 2 != 4, "was hoping against hope");

#endif // TT_ASSERT_ENABLE_DEBUG_ASSERTS
}
