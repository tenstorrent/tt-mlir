// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

namespace mlir::tt::d2m {

namespace gtest = ::testing;

using std::int32_t;
using std::int64_t;

// Test fixture for DstCapacityAnalysis tests.
struct DstCapacityAnalysisTest : public gtest::Test {
  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<func::FuncDialect, mlir::tt::d2m::D2MDialect>();
  }

  std::unique_ptr<MLIRContext> ctx;
};

// Test that DstCapacityAnalysis returns default capacity when no GenericOp is
// found.
TEST_F(DstCapacityAnalysisTest, NoGenericOpReturnsDefaultCapacity) {
  ctx->getOrLoadDialect<func::FuncDialect>();
  ctx->getOrLoadDialect<mlir::tt::d2m::D2MDialect>();

  OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();

  auto module = builder.create<ModuleOp>(loc);
  auto funcType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto funcOp =
      builder.create<func::FuncOp>(module.getLoc(), "test_func", funcType);
  funcOp.setPrivate();

  auto *block = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  builder.create<func::ReturnOp>(loc);

  DstCapacityAnalysis analysis(funcOp);
  EXPECT_EQ(analysis.getMinDstCapacity(), kDefaultDstCapacity);
}

// Test that DstCapacityAnalysis computes capacity correctly for a single
// GenericOp.
TEST_F(DstCapacityAnalysisTest, SingleGenericOpCapacity) {
  OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();

  auto module = builder.create<ModuleOp>(loc);
  auto funcType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto funcOp =
      builder.create<func::FuncOp>(module.getLoc(), "test_func", funcType);
  funcOp.setPrivate();

  auto *block = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  builder.create<func::ReturnOp>(loc);

  DstCapacityAnalysis analysis(funcOp);
  uint32_t capacity = analysis.getMinDstCapacity();
  EXPECT_EQ(capacity, kDefaultDstCapacity);
}

// Test that DstCapacityAnalysis returns minimum capacity across multiple
// GenericOps.
TEST_F(DstCapacityAnalysisTest, MultipleGenericOpsMinCapacity) {
  OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();

  auto module = builder.create<ModuleOp>(loc);
  auto funcType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto funcOp =
      builder.create<func::FuncOp>(module.getLoc(), "test_func", funcType);
  funcOp.setPrivate();

  auto *block = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  builder.create<func::ReturnOp>(loc);

  DstCapacityAnalysis analysis(funcOp);
  uint32_t capacity = analysis.getMinDstCapacity();
  EXPECT_EQ(capacity, kDefaultDstCapacity);
}

} // namespace mlir::tt::d2m
