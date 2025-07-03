// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/utils.h"
#include <gtest/gtest.h>

TEST(HandleFloat16ToBfloat16BufferCast, SanityFloat16ToBfloat16) {
  uint16_t float16_neg_one = 0xBC00;
  uint16_t bfloat16_neg_one = 0xBF80;
  uint16_t float_buffer[] = {float16_neg_one, 0, float16_neg_one};
  uint16_t bfloat_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleFloat16ToBFloat16(float_buffer,
                                                      bfloat_buffer, 3);

  EXPECT_EQ(bfloat_buffer[0], bfloat16_neg_one);
  EXPECT_EQ(bfloat_buffer[1], 0);
  EXPECT_EQ(bfloat_buffer[2], bfloat16_neg_one);
}

TEST(HandleFloat16ToBfloat16BufferCast, SanityBFloat16ToFloat16) {
  uint16_t float16_neg_one = 0xBC00;
  uint16_t bfloat16_neg_one = 0xBF80;
  uint16_t bfloat_buffer[] = {bfloat16_neg_one, 0, bfloat16_neg_one};
  uint16_t float_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleBFloat16ToFloat16(bfloat_buffer,
                                                      float_buffer, 3);

  EXPECT_EQ(float_buffer[0], float16_neg_one);
  EXPECT_EQ(float_buffer[1], 0);
  EXPECT_EQ(float_buffer[2], float16_neg_one);
}

TEST(HandleFloat16ToBfloat16BufferCast, HandleFloat16Max) {
  uint16_t bfloat16_from_float16_max = 0x4780; // = 65504.0f
  uint16_t bfloat16_from_float16_min = 0xc780; // = -65504.0f
  uint16_t float16_max = 0x7BFF;               // +65504.0
  uint16_t float16_min = 0xFBFF;               // -65504.0
  uint16_t float_buffer[] = {float16_min, float16_max};
  uint16_t bfloat_buffer[] = {0, 0};
  tt::runtime::utils::detail::handleFloat16ToBFloat16(float_buffer,
                                                      bfloat_buffer, 2);

  EXPECT_EQ(bfloat_buffer[0], bfloat16_from_float16_min);
  EXPECT_EQ(bfloat_buffer[1], bfloat16_from_float16_max);
}
