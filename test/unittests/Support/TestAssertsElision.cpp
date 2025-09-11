// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simulate builds with all asserts disabled.
#define TT_ASSERT_DISABLE_ASSERTS
#include "ttmlir/Asserts.h"

#include "testing/Utils.h"

TEST(AssertsDeathTest, MacroElisionAll) {
  // Both TT_assert* and TT_debug* should be elided
  // and not abort the process.

  TT_assert(2 < 1);
  TT_assert_limit(10, 1);
  TT_assertv(2 + 2 != 4, "was hoping against hope");

  TT_debug(2 < 1);
  TT_debug_limit(10, 1);
  TT_debugv(2 + 2 != 4, "was hoping against hope");
}

#undef TT_ASSERT_DISABLE_ASSERTS
