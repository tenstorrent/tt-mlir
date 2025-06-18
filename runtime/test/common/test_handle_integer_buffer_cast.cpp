// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/utils.h"
#include <gtest/gtest.h>
#include <limits>

TEST(HandleI64ToI32BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  int32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the int32_t range, not overflowed
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int32_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int32_t>::max());
}

TEST(HandleI64ToI16BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  int16_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the int16_t range, not overflowed
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int16_t>::max());
}

TEST(HandleI64ToI8BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  int8_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the int8_t range, not overflowed
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI64ToUI64BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  uint64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint64_t range, not overflowed
  // uint64_t has a larger max value than int64_t, so the int64_t max value
  // should remain
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int64_t>::max());
}

TEST(HandleI64ToUI32BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  uint32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint32_t range, not overflowed
  // uint32_t has a smaller max value than int64_t, so the int64_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint32_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<uint32_t>::max());
}

TEST(HandleI64ToUI16BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  uint16_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint16_t range, not overflowed
  // uint16_t has a smaller max value than int64_t, so the int64_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<uint16_t>::max());
}

TEST(HandleI64ToUI8BufferCast, Sanity) {
  int64_t old_buffer[] = {std::numeric_limits<int64_t>::min(), 0,
                          std::numeric_limits<int64_t>::max()};
  uint8_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint8_t range, not overflowed
  // uint8_t has a smaller max value than int64_t, so the int64_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<uint8_t>::max());
}

TEST(HandleI32ToI64BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  int64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be the same in the int64 buffer as int64 has larger bitwidth
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int32_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int32_t>::max());
}

TEST(HandleI32ToI16BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  int16_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the int16_t range, not overflowed
  // int16_t has a smaller max value than int32_t, so the int32_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int16_t>::max());
}

TEST(HandleI32ToI8BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  int8_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the int8_t range, not overflowed
  // int8_t has a smaller max value than int32_t, so the int32_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI32ToUI64BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  uint64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint64_t range, not overflowed
  // uint64_t has a larger max value than int32_t, so the int32_t max value
  // should remain
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int32_t>::max());
}

TEST(HandleI32ToUI32BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  uint32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint32_t range, not overflowed
  // uint32_t has a larger max value than int32_t, so the int32_t max value
  // should remain
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint32_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int32_t>::max());
}

TEST(HandleI32ToUI16BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  uint16_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint16_t range, not overflowed
  // uint16_t has a smaller max value than int32_t, so the int32_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<uint16_t>::max());
}

TEST(HandleI32ToUI8BufferCast, Sanity) {
  int32_t old_buffer[] = {std::numeric_limits<int32_t>::min(), 0,
                          std::numeric_limits<int32_t>::max()};
  uint8_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint8_t range, not overflowed
  // uint8_t has a smaller max value than int32_t, so the int32_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<uint8_t>::max());
}

TEST(HandleI16ToI64BufferCast, Sanity) {
  int16_t old_buffer[] = {std::numeric_limits<int16_t>::min(), 0,
                          std::numeric_limits<int16_t>::max()};
  int64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be the same in the int64 buffer as int64 has larger bitwidth
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int16_t>::max());
}

TEST(HandleI16ToI32BufferCast, Sanity) {
  int16_t old_buffer[] = {std::numeric_limits<int16_t>::min(), 0,
                          std::numeric_limits<int16_t>::max()};
  int32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be the same in the int32 buffer as int32 has larger bitwidth
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int16_t>::max());
}

TEST(HandleI16ToI8BufferCast, Sanity) {
  int16_t old_buffer[] = {std::numeric_limits<int16_t>::min(), 0,
                          std::numeric_limits<int16_t>::max()};
  int8_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the int8_t range, not overflowed
  // int8_t has a smaller max value than int16_t, so the int16_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI16ToUI64BufferCast, Sanity) {
  int16_t old_buffer[] = {std::numeric_limits<int16_t>::min(), 0,
                          std::numeric_limits<int16_t>::max()};
  uint64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint64_t range, not overflowed
  // uint64_t has a larger max value than int16_t, so the int16_t max value
  // should remain
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int16_t>::max());
}

TEST(HandleI16ToUI32BufferCast, Sanity) {
  int16_t old_buffer[] = {std::numeric_limits<int16_t>::min(), 0,
                          std::numeric_limits<int16_t>::max()};
  uint32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint32_t range, not overflowed
  // uint32_t has a larger max value than int16_t, so the int16_t max value
  // should remain
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint32_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int16_t>::max());
}

TEST(HandleI16ToUI8BufferCast, Sanity) {
  int16_t old_buffer[] = {std::numeric_limits<int16_t>::min(), 0,
                          std::numeric_limits<int16_t>::max()};
  uint8_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint8_t range, not overflowed
  // uint8_t has a smaller max value than int16_t, so the int16_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<uint8_t>::max());
}

TEST(HandleI8ToI64BufferCast, Sanity) {
  int8_t old_buffer[] = {std::numeric_limits<int8_t>::min(), 0,
                         std::numeric_limits<int8_t>::max()};
  int64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be the same in the int64 buffer as int64 has larger bitwidth
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI8ToI32BufferCast, Sanity) {
  int8_t old_buffer[] = {std::numeric_limits<int8_t>::min(), 0,
                         std::numeric_limits<int8_t>::max()};
  int32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be the same in the int32 buffer as int32 has larger bitwidth
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI8ToI16BufferCast, Sanity) {
  int8_t old_buffer[] = {std::numeric_limits<int8_t>::min(), 0,
                         std::numeric_limits<int8_t>::max()};
  int16_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be the same in the int16 buffer as int16 has larger bitwidth
  EXPECT_EQ(new_buffer[0], std::numeric_limits<int8_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI8ToUI64BufferCast, Sanity) {
  int8_t old_buffer[] = {std::numeric_limits<int8_t>::min(), 0,
                         std::numeric_limits<int8_t>::max()};
  uint64_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint64_t range, not overflowed
  // uint64_t has a larger max value than int8_t, so the int8_t max value should
  // remain
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint64_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI8ToUI32BufferCast, Sanity) {
  int8_t old_buffer[] = {std::numeric_limits<int8_t>::min(), 0,
                         std::numeric_limits<int8_t>::max()};
  uint32_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint32_t range, not overflowed
  // uint32_t has a smaller max value than int8_t, so the int8_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint32_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}

TEST(HandleI8ToUI16BufferCast, Sanity) {
  int8_t old_buffer[] = {std::numeric_limits<int8_t>::min(), 0,
                         std::numeric_limits<int8_t>::max()};
  uint16_t new_buffer[] = {0, 0, 0};
  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      3);

  // Values should be clamped to the uint16_t range, not overflowed
  // uint16_t has a smaller max value than int8_t, so the int8_t max value
  // should be clamped
  EXPECT_EQ(new_buffer[0], std::numeric_limits<uint16_t>::min());
  EXPECT_EQ(new_buffer[1], 0);
  EXPECT_EQ(new_buffer[2], std::numeric_limits<int8_t>::max());
}
