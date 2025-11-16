// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for DstCapacityAnalysis.
//
// Note: Full capacity computation testing (f16→16 tiles, f32→8 tiles) requires
// a registered chip descriptor and GenericOp operations. This is covered by
// integration tests in:
//   - test/ttmlir/Dialect/D2M/Transforms/dst_graph_coloring/*.mlir
//
// These unit tests verify the default behavior when no GenericOp is present.

#include "ttmlir/Dialect/D2M/Analysis/DstCapacityAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

namespace mlir::tt::d2m {

namespace gtest = ::testing;

using std::int32_t;
using std::int64_t;

// Test fixture for DstCapacityAnalysis tests.
struct DstCapacityAnalysisTest : public gtest::Test {
  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<func::FuncDialect, mlir::tt::d2m::D2MDialect,
                     mlir::tt::ttcore::TTCoreDialect, affine::AffineDialect,
                     arith::ArithDialect>();
  }

  std::unique_ptr<MLIRContext> ctx;
};

// Test that DstCapacityAnalysis returns default capacity when no GenericOp is
// found.
TEST_F(DstCapacityAnalysisTest, NoGenericOpReturnsDefaultCapacity) {
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

  // Test with default fullSyncEn=true
  DstCapacityAnalysis analysis(funcOp);
  EXPECT_EQ(analysis.getMinDstCapacity(), kDefaultDstCapacity);
}

// Test that DstCapacityAnalysis returns default capacity when there are
// acquire_dst/release_dst operations but no GenericOp.
TEST_F(DstCapacityAnalysisTest, NoGenericOpWithDstOperations) {
  const char *moduleStr = R"(
    #dst_ = #ttcore.memory_space<dst>
    module {
      func.func @test_func() {
        %dst = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst_>
        affine.for %i = 0 to 1 {
          affine.for %j = 0 to 1 {
            %val = affine.load %dst[0, %i, %j] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst_>
            affine.store %val, %dst[1, %i, %j] : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst_>
          }
        }
        d2m.release_dst %dst : memref<8x1x1x!ttcore.tile<32x32, f32>, #dst_>
        return
      }
    }
  )";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, ctx.get());
  ASSERT_TRUE(module);

  auto funcOp = module->lookupSymbol<func::FuncOp>("test_func");
  ASSERT_TRUE(funcOp);

  // Without GenericOp, the analysis returns default capacity.
  // The analysis only examines GenericOp compute regions.
  // Test with default fullSyncEn=true
  DstCapacityAnalysis analysis(funcOp);
  EXPECT_EQ(analysis.getMinDstCapacity(), kDefaultDstCapacity);
}

// Test that DstCapacityAnalysis handles empty functions correctly.
TEST_F(DstCapacityAnalysisTest, EmptyFunction) {
  OpBuilder builder(ctx.get());
  auto loc = builder.getUnknownLoc();

  auto module = builder.create<ModuleOp>(loc);
  auto funcType = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
  auto funcOp =
      builder.create<func::FuncOp>(module.getLoc(), "empty_func", funcType);
  funcOp.setPrivate();

  auto *block = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  builder.create<func::ReturnOp>(loc);

  // Test with default fullSyncEn=true
  DstCapacityAnalysis analysis(funcOp);
  EXPECT_EQ(analysis.getMinDstCapacity(), kDefaultDstCapacity);
}

} // namespace mlir::tt::d2m
