// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/utils.h"
#include <gtest/gtest.h>
#include <limits>

TEST(HandleFloat16ToBfloat16BufferCast, Sanity) {
  uint16_t float16_neg_one = 0xBC00; 
  uint16_t bfloat16_neg_one = 0xBF80; 
  uint16_t float_buffer[] = {float16_neg_one, 0, float16_neg_one};
  uint16_t bfloat_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleFloat16ToBFloat16(float_buffer, bfloat_buffer,
                                                      3);

  // Values should be clamped to the int32_t range, not overflowed
  EXPECT_EQ(bfloat_buffer[0], bfloat16_neg_one);
  EXPECT_EQ(bfloat_buffer[1], 0);
  EXPECT_EQ(bfloat_buffer[2], bfloat16_neg_one);
}