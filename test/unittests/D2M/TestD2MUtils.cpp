#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::tt;

class D2MUtilsTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> ctx;

  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<func::FuncDialect, d2m::D2MDialect, ttcore::TTCoreDialect,
                     affine::AffineDialect, memref::MemRefDialect,
                     arith::ArithDialect>();
  }

  func::FuncOp createTestFunction(OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_func", funcType);
    funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(&funcOp.getBody().front());
    return funcOp;
  }
};

TEST_F(D2MUtilsTest, GetRegionLargestDstElemTypeEmptyRegion) {
  OpBuilder builder(ctx.get());
  auto funcOp = createTestFunction(builder);
  Region &region = funcOp.getBody();

  // Empty region should return default f32 type.
  Type largestType = d2m::utils::getRegionLargestDstElemType(region);
  ASSERT_TRUE(mlir::isa<FloatType>(largestType));
  EXPECT_EQ(mlir::cast<FloatType>(largestType).getWidth(), 32u);
}

TEST_F(D2MUtilsTest, GetSquareTargetGrid) {
  // Test that getSquareTargetGrid returns a square grid where all dimensions
  // are set to the minimum value.
  auto grid1 = d2m::utils::getSquareTargetGrid({2, 3});
  EXPECT_EQ(grid1.size(), 2u);
  EXPECT_EQ(grid1[0], 2);
  EXPECT_EQ(grid1[1], 2);

  auto grid2 = d2m::utils::getSquareTargetGrid({5, 3, 4});
  EXPECT_EQ(grid2.size(), 3u);
  EXPECT_EQ(grid2[0], 3);
  EXPECT_EQ(grid2[1], 3);
  EXPECT_EQ(grid2[2], 3);

  auto grid3 = d2m::utils::getSquareTargetGrid({1, 1});
  EXPECT_EQ(grid3.size(), 2u);
  EXPECT_EQ(grid3[0], 1);
  EXPECT_EQ(grid3[1], 1);
}
