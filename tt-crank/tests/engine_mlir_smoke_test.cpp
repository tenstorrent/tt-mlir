#include "engine/mlir_smoke.hpp"

#include <gtest/gtest.h>

TEST(MlirSmokeTest, RegistersTtirPipelines) {
    EXPECT_TRUE(tt::kurbla::mlir_smoke());
}
