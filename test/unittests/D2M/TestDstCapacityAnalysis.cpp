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

// Test that DstCapacityAnalysis correctly computes capacity for f16 operations.
// This test enables debug output and verifies that the analysis reports the
// expected largest DST type (f16) and minimum capacity (16 tiles with fullSyncEn=true).
TEST_F(DstCapacityAnalysisTest, F16CapacityAnalysisDebugOutput) {
  const char *moduleStr = R"(
    #l1_ = #ttcore.memory_space<l1>
    #dst_ = #ttcore.memory_space<dst>
    #system_desc = #ttcore.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1073741824, pcie_address = 0, noc_l1_address_align_bytes = 16, noc_dram_address_align_bytes = 32, noc0_l1_address_offset = 0, noc0_dram_address_offset = 0, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824}]>
    module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device {mesh = #ttcore.mesh<{}>, chip = [{arch = "wormhole_b0", subchip_index = 0}]}
      func.func @test_f16_capacity() {
        %in0 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in1 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in2 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in3 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in4 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in5 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in6 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in7 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in8 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in9 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in10 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in11 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in12 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in13 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in14 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in15 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>

        d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<compute>]
        } ins(%in0, %in1, %in2, %in3, %in4, %in5, %in6, %in7, %in8, %in9, %in10, %in11, %in12, %in13, %in14, %in15 :
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
          outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
        ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb3: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb4: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb5: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb6: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb7: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb8: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb9: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb10: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb11: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb12: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb13: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb14: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb15: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
          %c0 = arith.constant 0 : index

          %0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %2 = d2m.wait %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %3 = d2m.wait %cb3 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %4 = d2m.wait %cb4 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %5 = d2m.wait %cb5 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %6 = d2m.wait %cb6 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %7 = d2m.wait %cb7 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %8 = d2m.wait %cb8 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %9 = d2m.wait %cb9 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %10 = d2m.wait %cb10 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %11 = d2m.wait %cb11 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %12 = d2m.wait %cb12 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %13 = d2m.wait %cb13 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %14 = d2m.wait %cb14 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %15 = d2m.wait %cb15 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %out_mem = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

          // Load 16 values into DST - these will all need DST registers and interfere.
          %v0 = affine.load %0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v1 = affine.load %1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v2 = affine.load %2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v3 = affine.load %3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v4 = affine.load %4[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v5 = affine.load %5[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v6 = affine.load %6[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v7 = affine.load %7[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v8 = affine.load %8[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v9 = affine.load %9[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v10 = affine.load %10[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v11 = affine.load %11[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v12 = affine.load %12[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v13 = affine.load %13[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v14 = affine.load %14[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v15 = affine.load %15[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

          // Add them all together to force all 16 to be live simultaneously.
          %r0 = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r1 = "d2m.tile_add"(%r0, %v2) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r2 = "d2m.tile_add"(%r1, %v3) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r3 = "d2m.tile_add"(%r2, %v4) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r4 = "d2m.tile_add"(%r3, %v5) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r5 = "d2m.tile_add"(%r4, %v6) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r6 = "d2m.tile_add"(%r5, %v7) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r7 = "d2m.tile_add"(%r6, %v8) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r8 = "d2m.tile_add"(%r7, %v9) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r9 = "d2m.tile_add"(%r8, %v10) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r10 = "d2m.tile_add"(%r9, %v11) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r11 = "d2m.tile_add"(%r10, %v12) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r12 = "d2m.tile_add"(%r11, %v13) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r13 = "d2m.tile_add"(%r12, %v14) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %result = "d2m.tile_add"(%r13, %v15) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>

          affine.store %result, %out_mem[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
        }
        return
      }
    }
  )";

  // Enable debug output for dst-capacity-analysis
  llvm::DebugFlag = true;
  llvm::setCurrentDebugType("dst-capacity-analysis");

  // Start capturing stderr to get the debug output
  testing::internal::CaptureStderr();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, ctx.get());
  ASSERT_TRUE(module);

  auto funcOp = module->lookupSymbol<func::FuncOp>("test_f16_capacity");
  ASSERT_TRUE(funcOp);

  // Run the analysis with fullSyncEn=true (default) - this should trigger the debug output
  DstCapacityAnalysis analysis(funcOp, /*fullSyncEn=*/true);

  // Get the captured debug output
  std::string debugOutput = testing::internal::GetCapturedStderr();

  // Verify the expected capacity (16 for f16 with fullSyncEn=true)
  EXPECT_EQ(analysis.getMinDstCapacity(), 16u);

  // Verify debug output contains the expected information
  EXPECT_TRUE(
      debugOutput.find("DST Capacity Analysis: Largest DST type: f16") !=
      std::string::npos);
  EXPECT_TRUE(debugOutput.find("fullSyncEn: 1") != std::string::npos);
  EXPECT_TRUE(debugOutput.find("Capacity: 16") != std::string::npos);

  // Disable debug output
  llvm::DebugFlag = false;
}

// Test that graph coloring fails when capacity is exceeded.
// This extends the previous test with 17 concurrent DST accesses.
TEST_F(DstCapacityAnalysisTest, F16CapacityExceededError) {
  const char *moduleStr = R"(
    #l1_ = #ttcore.memory_space<l1>
    #dst_ = #ttcore.memory_space<dst>
    #system_desc = #ttcore.system_desc<[{arch = <wormhole_b0>, grid = 8x8, l1_size = 1048576, num_dram_channels = 12, dram_channel_size = 1073741824, pcie_address = 0, noc_l1_address_align_bytes = 16, noc_dram_address_align_bytes = 32, noc0_l1_address_offset = 0, noc0_dram_address_offset = 0, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824}]>
    module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device {mesh = #ttcore.mesh<{}>, chip = [{arch = "wormhole_b0", subchip_index = 0}]}
      func.func @test_f16_capacity_exceeded() {
        %in0 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in1 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in2 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in3 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in4 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in5 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in6 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in7 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in8 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in9 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in10 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in11 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in12 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in13 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in14 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in15 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %in16 = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>
        %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>

        d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<compute>]
        } ins(%in0, %in1, %in2, %in3, %in4, %in5, %in6, %in7, %in8, %in9, %in10, %in11, %in12, %in13, %in14, %in15, %in16 :
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>,
              memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>)
          outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f16>, #ttcore.shard<4096x4096>, #l1_>) {
        ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb3: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb4: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb5: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb6: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb7: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb8: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb9: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb10: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb11: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb12: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb13: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb14: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb15: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb16: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>,
                  %cb_out: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>>):
          %c0 = arith.constant 0 : index

          %0 = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %1 = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %2 = d2m.wait %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %3 = d2m.wait %cb3 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %4 = d2m.wait %cb4 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %5 = d2m.wait %cb5 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %6 = d2m.wait %cb6 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %7 = d2m.wait %cb7 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %8 = d2m.wait %cb8 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %9 = d2m.wait %cb9 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %10 = d2m.wait %cb10 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %11 = d2m.wait %cb11 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %12 = d2m.wait %cb12 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %13 = d2m.wait %cb13 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %14 = d2m.wait %cb14 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %15 = d2m.wait %cb15 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %16 = d2m.wait %cb16 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %out_mem = d2m.reserve %cb_out : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f16>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

          // Load 17 values into DST - this exceeds f16 capacity (16 tiles).
          %v0 = affine.load %0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v1 = affine.load %1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v2 = affine.load %2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v3 = affine.load %3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v4 = affine.load %4[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v5 = affine.load %5[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v6 = affine.load %6[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v7 = affine.load %7[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v8 = affine.load %8[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v9 = affine.load %9[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v10 = affine.load %10[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v11 = affine.load %11[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v12 = affine.load %12[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v13 = affine.load %13[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v14 = affine.load %14[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v15 = affine.load %15[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
          %v16 = affine.load %16[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>

          // Add them all together - this should trigger the capacity error.
          %r0 = "d2m.tile_add"(%v0, %v1) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r1 = "d2m.tile_add"(%r0, %v2) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r2 = "d2m.tile_add"(%r1, %v3) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r3 = "d2m.tile_add"(%r2, %v4) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r4 = "d2m.tile_add"(%r3, %v5) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r5 = "d2m.tile_add"(%r4, %v6) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r6 = "d2m.tile_add"(%r5, %v7) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r7 = "d2m.tile_add"(%r6, %v8) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r8 = "d2m.tile_add"(%r7, %v9) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r9 = "d2m.tile_add"(%r8, %v10) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r10 = "d2m.tile_add"(%r9, %v11) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r11 = "d2m.tile_add"(%r10, %v12) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r12 = "d2m.tile_add"(%r11, %v13) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r13 = "d2m.tile_add"(%r12, %v14) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %r14 = "d2m.tile_add"(%r13, %v15) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>
          %result = "d2m.tile_add"(%r14, %v16) : (!ttcore.tile<32x32, f16>, !ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f16>

          affine.store %result, %out_mem[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f16>, #l1_>
        }
        return
      }
    }
  )";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, ctx.get());
  ASSERT_TRUE(module);

  auto funcOp =
      module->lookupSymbol<func::FuncOp>("test_f16_capacity_exceeded");
  ASSERT_TRUE(funcOp);

  // Verify the analysis computes the correct capacity (16 for f16 with fullSyncEn=true)
  DstCapacityAnalysis analysis(funcOp, /*fullSyncEn=*/true);
  EXPECT_EQ(analysis.getMinDstCapacity(), 16u);

  // TODO: Add integration test using PassManager to verify the pass fails
  // when capacity is exceeded. This requires running the full pass pipeline
  // which is better suited for lit tests.
}

} // namespace mlir::tt::d2m
