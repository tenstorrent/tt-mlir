// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "version.hpp"

#include <gtest/gtest.h>

TEST(VersionTest, IsNonEmpty) {
    EXPECT_FALSE(tt::kurbla::version().empty());
}
