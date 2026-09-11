#include "version.hpp"

#include <gtest/gtest.h>

TEST(VersionTest, IsNonEmpty) {
    EXPECT_FALSE(tt::kurbla::version().empty());
}
