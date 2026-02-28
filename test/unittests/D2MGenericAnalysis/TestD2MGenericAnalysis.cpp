// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

namespace mlir::tt::d2m {

class GenericOpAnalysisTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  mlir::AffineExpr d0, d1, d2, d3;

  GenericOpAnalysisTest() {
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::linalg::LinalgDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::d2m::D2MDialect>();

    d0 = mlir::getAffineDimExpr(0, &context);
    d1 = mlir::getAffineDimExpr(1, &context);
    d2 = mlir::getAffineDimExpr(2, &context);
    d3 = mlir::getAffineDimExpr(3, &context);
  }

  mlir::AffineMap makeAffineMap(unsigned dimCount,
                                mlir::ArrayRef<mlir::AffineExpr> exprs) {
    return mlir::AffineMap::get(dimCount, 0, exprs, &context);
  }

  void checkDimConstraints(
      std::vector<std::pair<mlir::AffineMap, mlir::SmallVector<int64_t>>>
          operands,
      std::optional<mlir::SmallVector<int64_t>> expected) {
    mlir::SmallVector<mlir::AffineMap> maps;
    mlir::SmallVector<mlir::SmallVector<int64_t>> shapes;
    for (const auto &op : operands) {
      maps.push_back(op.first);
      shapes.push_back(op.second);
    }
    auto result = d2m::utils::computeDimConstraints(maps, shapes);
    EXPECT_EQ(result, expected);
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseModule(llvm::StringRef moduleText) {
    mlir::OwningOpRef<mlir::Operation *> parsedOp =
        mlir::parseSourceString<mlir::Operation *>(moduleText, &context);
    if (!parsedOp) {
      return {};
    }
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(*parsedOp)) {
      return mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::cast<mlir::ModuleOp>(parsedOp.release()));
    }

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    module->push_back(parsedOp.release());
    return module;
  }

  d2m::GenericOp getSingleGenericOp(mlir::ModuleOp module) {
    d2m::GenericOp genericOp;
    module.walk([&](d2m::GenericOp op) { genericOp = op; });
    return genericOp;
  }

  void markAllForLoopsAsBlocking(d2m::GenericOp genericOp) {
    genericOp->walk([&](mlir::scf::ForOp forOp) {
      forOp->setAttr("d2m.blocking_loop", mlir::UnitAttr::get(&context));
    });
  }
};

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsEltwise) {
  // 2D shape
  checkDimConstraints({{makeAffineMap(2, {d0, d1}), {1, 2}},
                       {makeAffineMap(2, {d0, d1}), {1, 2}}},
                      std::make_optional(mlir::SmallVector<int64_t>{1, 2}));

  // 3D shape
  checkDimConstraints({{makeAffineMap(3, {d0, d1, d2}), {1, 2, 4}},
                       {makeAffineMap(3, {d0, d1, d2}), {1, 2, 4}}},
                      std::make_optional(mlir::SmallVector<int64_t>{1, 2, 4}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsForPermute) {
  // 2D transpose
  checkDimConstraints({{makeAffineMap(2, {d0, d1}), {1, 2}},
                       {makeAffineMap(2, {d1, d0}), {2, 1}}},
                      std::make_optional(mlir::SmallVector<int64_t>{1, 2}));

  // 3D permute
  checkDimConstraints(
      {
          {makeAffineMap(3, {d2, d1, d0}), {3, 2, 1}},
          {makeAffineMap(3, {d0, d1, d2}), {1, 2, 3}},
      },
      std::make_optional(mlir::SmallVector<int64_t>{1, 2, 3}));

  // 4D permute with multiple dim swaps
  checkDimConstraints(
      {
          {makeAffineMap(4, {d0, d1, d2, d3}), {4, 3, 2, 5}},
          {makeAffineMap(4, {d2, d3, d0, d1}), {2, 5, 4, 3}},
      },
      std::make_optional(mlir::SmallVector<int64_t>{4, 3, 2, 5}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsForMatmul) {
  // dim mapping: d0 = M, d1 = N, d2 = K
  checkDimConstraints({{makeAffineMap(3, {d0, d2}), {3, 5}},
                       {makeAffineMap(3, {d2, d1}), {5, 2}},
                       {makeAffineMap(3, {d0, d1}), {3, 2}}},
                      std::make_optional(mlir::SmallVector<int64_t>{3, 2, 5}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzePartialDimConstraintsForMatmul) {
  // only output shape is provided; d2 is a 'free dim' and not constrained.
  checkDimConstraints({{makeAffineMap(3, {d0, d1}), {3, 2}}},
                      std::make_optional(mlir::SmallVector<int64_t>{3, 2, 0}));
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDimConstraintsForIncompatibleShapes) {
  checkDimConstraints({{makeAffineMap(2, {d0, d1}), {1, 2}},
                       {makeAffineMap(2, {d0, d1}), {1, 3}}},
                      std::nullopt);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingSFPU) {
  constexpr llvm::StringLiteral moduleText = R"mlir(
func.func @test(
    %in: memref<1x1x8x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x8x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in : memref<1x1x8x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x8x!ttcore.tile<32x32, f32>>) {
    ^unified0(%cb0: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>,
              %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>):
      %in_m = d2m.wait %cb0 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %out_m = d2m.reserve %cb1 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%in_m : memref<1x8x!ttcore.tile<32x32, f32>>)
          outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
        ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
          %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %abs : !ttcore.tile<32x32, f32>
        }
      }
    }
  return
}
)mlir";

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  auto packing = d2m::utils::analyzeGenericForDSTPacking(generic);
  ASSERT_EQ(packing.size(), 1u);
  EXPECT_EQ(packing.front().second.num_tiles_per_flip, 2);
  EXPECT_EQ(packing.front().second.num_dst_flips, 2);
  EXPECT_EQ(packing.front().second.num_outer_loop_iters, 2);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingFPU) {
  constexpr llvm::StringLiteral moduleText = R"mlir(
func.func @test(
    %in0: memref<1x1x8x!ttcore.tile<32x32, f32>>,
    %in1: memref<1x1x8x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x8x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in0, %in1 : memref<1x1x8x!ttcore.tile<32x32, f32>>, memref<1x1x8x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x8x!ttcore.tile<32x32, f32>>) {
    ^unified0(%cb0: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>,
              %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>,
              %cb2: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>):
      %in0_m = d2m.wait %cb0 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %in1_m = d2m.wait %cb1 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %out_m = d2m.reserve %cb2 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%in0_m, %in1_m : memref<1x8x!ttcore.tile<32x32, f32>>, memref<1x8x!ttcore.tile<32x32, f32>>)
          outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
        ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
          %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %sum : !ttcore.tile<32x32, f32>
        }
      }
    }
  return
}
)mlir";

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  auto packing = d2m::utils::analyzeGenericForDSTPacking(generic);
  ASSERT_EQ(packing.size(), 1u);
  EXPECT_EQ(packing.front().second.num_tiles_per_flip, 4);
  EXPECT_EQ(packing.front().second.num_dst_flips, 2);
  EXPECT_EQ(packing.front().second.num_outer_loop_iters, 1);
}

TEST_F(GenericOpAnalysisTest, RejectsDifferentImmediateParentBlockingLoops) {
  constexpr llvm::StringLiteral moduleText = R"mlir(
func.func @test(
    %in: memref<1x1x8x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x8x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in : memref<1x1x8x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x8x!ttcore.tile<32x32, f32>>) {
    ^unified0(%cb0: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>,
              %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>):
      %in_m = d2m.wait %cb0 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %out_m = d2m.reserve %cb1 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%in_m : memref<1x8x!ttcore.tile<32x32, f32>>)
          outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
        ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
          %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %abs : !ttcore.tile<32x32, f32>
        }
      }
      scf.for %j = %c0 to %c8 step %c1 {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%in_m : memref<1x8x!ttcore.tile<32x32, f32>>)
          outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
        ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
          %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %abs : !ttcore.tile<32x32, f32>
        }
      }
    }
  return
}
)mlir";

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  auto packing = d2m::utils::analyzeGenericForDSTPacking(generic);
  EXPECT_TRUE(packing.empty());
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingManyMixedOps) {
  constexpr llvm::StringLiteral moduleText = R"mlir(
func.func @test(
    %in0: memref<1x1x8x!ttcore.tile<32x32, f32>>,
    %in1: memref<1x1x8x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x8x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in0, %in1 : memref<1x1x8x!ttcore.tile<32x32, f32>>, memref<1x1x8x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x8x!ttcore.tile<32x32, f32>>) {
  ^unified0(%cb0: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>,
            %cb1: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>,
            %cb2: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>>):
    %in0_m = d2m.wait %cb0 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
    %in1_m = d2m.wait %cb1 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
    %out_m = d2m.reserve %cb2 : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>>> -> memref<1x8x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x8x!ttcore.tile<32x32, f32>>, memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sum : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %cst = arith.constant 2.0 : f32
        %mul = "d2m.tile_mul"(%arg0, %cst) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>
        linalg.yield %mul : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x8x!ttcore.tile<32x32, f32>>, memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sub = "d2m.tile_sub"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sub : !ttcore.tile<32x32, f32>
      }
    }
  }
  return
}
)mlir";

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  auto packing = d2m::utils::analyzeGenericForDSTPacking(generic);
  ASSERT_EQ(packing.size(), 4u);
  EXPECT_EQ(packing[0].second.num_tiles_per_flip, 2); // tile_abs => SFPU fp32
  EXPECT_EQ(packing[1].second.num_tiles_per_flip,
            4); // tile_add(tile,tile) => FPU fp32
  EXPECT_EQ(packing[2].second.num_tiles_per_flip,
            2); // tile_mul(tile,scalar) => SFPU fp32
  EXPECT_EQ(packing[3].second.num_tiles_per_flip,
            4); // tile_sub(tile,tile) => FPU fp32
  EXPECT_EQ(packing[0].second.num_dst_flips, 4);
  EXPECT_EQ(packing[1].second.num_dst_flips, 2);
  EXPECT_EQ(packing[2].second.num_dst_flips, 4);
  EXPECT_EQ(packing[3].second.num_dst_flips, 2);
  EXPECT_EQ(packing[0].second.num_outer_loop_iters, 1);
  EXPECT_EQ(packing[1].second.num_outer_loop_iters, 1);
  EXPECT_EQ(packing[2].second.num_outer_loop_iters, 1);
  EXPECT_EQ(packing[3].second.num_outer_loop_iters, 1);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingPrimeShardShapes) {
  constexpr llvm::StringLiteral moduleText = R"mlir(
func.func @test(
    %in0: memref<1x1x7x!ttcore.tile<32x32, f32>>,
    %in1: memref<1x1x7x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x7x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in0, %in1 : memref<1x1x7x!ttcore.tile<32x32, f32>>, memref<1x1x7x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x7x!ttcore.tile<32x32, f32>>) {
  ^unified0(%cb0: !d2m.cb<memref<1x7x!ttcore.tile<32x32, f32>>>,
            %cb1: !d2m.cb<memref<1x7x!ttcore.tile<32x32, f32>>>,
            %cb2: !d2m.cb<memref<1x7x!ttcore.tile<32x32, f32>>>):
    %in0_m = d2m.wait %cb0 : !d2m.cb<memref<1x7x!ttcore.tile<32x32, f32>>> -> memref<1x7x!ttcore.tile<32x32, f32>>
    %in1_m = d2m.wait %cb1 : !d2m.cb<memref<1x7x!ttcore.tile<32x32, f32>>> -> memref<1x7x!ttcore.tile<32x32, f32>>
    %out_m = d2m.reserve %cb2 : !d2m.cb<memref<1x7x!ttcore.tile<32x32, f32>>> -> memref<1x7x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    scf.for %i = %c0 to %c7 step %c1 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x7x!ttcore.tile<32x32, f32>>, memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sum : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %cst = arith.constant 3.0 : f32
        %mul = "d2m.tile_mul"(%arg0, %cst) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>
        linalg.yield %mul : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x7x!ttcore.tile<32x32, f32>>, memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sub = "d2m.tile_sub"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sub : !ttcore.tile<32x32, f32>
      }
    }
  }
  return
}
)mlir";

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  auto packing = d2m::utils::analyzeGenericForDSTPacking(generic);
  ASSERT_EQ(packing.size(), 4u);
  EXPECT_EQ(packing[0].second.num_tiles_per_flip, 1); // prime shard, SFPU fp32
  EXPECT_EQ(packing[1].second.num_tiles_per_flip, 1); // prime shard, FPU fp32
  EXPECT_EQ(packing[2].second.num_tiles_per_flip, 1); // prime shard, SFPU fp32
  EXPECT_EQ(packing[3].second.num_tiles_per_flip, 1); // prime shard, FPU fp32
  EXPECT_EQ(packing[0].second.num_dst_flips, 7);
  EXPECT_EQ(packing[1].second.num_dst_flips, 7);
  EXPECT_EQ(packing[2].second.num_dst_flips, 7);
  EXPECT_EQ(packing[3].second.num_dst_flips, 7);
  EXPECT_EQ(packing[0].second.num_outer_loop_iters, 1);
  EXPECT_EQ(packing[1].second.num_outer_loop_iters, 1);
  EXPECT_EQ(packing[2].second.num_outer_loop_iters, 1);
  EXPECT_EQ(packing[3].second.num_outer_loop_iters, 1);
}

} // namespace mlir::tt::d2m
