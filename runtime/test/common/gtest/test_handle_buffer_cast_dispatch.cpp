// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Regression coverage for the handleBufferCast() dispatcher (the DataType
// switch), specifically the signedness-mismatch readback pairs that the
// same-width-integer alias relaxation in ttnn::memcpy() now permits. ttnn ops
// such as argmax return UInt32 indices that the host reads back into an Int64
// buffer (Int64's supported alias is Int32). Without the UInt32 -> Int64 (and
// symmetric Int32 -> UInt64) branches the dispatcher would throw.
#include "tt/runtime/utils.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

TEST(HandleBufferCastDispatch, UInt32ToInt64Widens) {
  std::vector<uint32_t> src = {0u, 1u, 42u, 128255u,
                               std::numeric_limits<uint32_t>::max()};
  std::vector<int64_t> dst(src.size(), 0);

  EXPECT_NO_THROW(tt::runtime::utils::handleBufferCast(
      src.data(), dst.data(), ::tt::target::DataType::UInt32,
      ::tt::target::DataType::Int64, src.size()));

  for (size_t i = 0; i < src.size(); ++i) {
    EXPECT_EQ(dst[i], static_cast<int64_t>(src[i])) << "element " << i;
  }
}

TEST(HandleBufferCastDispatch, Int32ToUInt64Widens) {
  std::vector<int32_t> src = {0, 1, 42, 128255,
                              std::numeric_limits<int32_t>::max()};
  std::vector<uint64_t> dst(src.size(), 0);

  EXPECT_NO_THROW(tt::runtime::utils::handleBufferCast(
      src.data(), dst.data(), ::tt::target::DataType::Int32,
      ::tt::target::DataType::UInt64, src.size()));

  for (size_t i = 0; i < src.size(); ++i) {
    EXPECT_EQ(dst[i], static_cast<uint64_t>(src[i])) << "element " << i;
  }
}

} // namespace
