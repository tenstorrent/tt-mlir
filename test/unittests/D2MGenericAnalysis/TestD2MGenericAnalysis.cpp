// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

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
      auto owningModule = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::cast<mlir::ModuleOp>(parsedOp.release()));
      if (failed(runRegisterDevicePass(*owningModule))) {
        return {};
      }
      return owningModule;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    module->push_back(parsedOp.release());
    if (failed(runRegisterDevicePass(*module))) {
      return {};
    }
    return module;
  }

  mlir::LogicalResult runRegisterDevicePass(mlir::ModuleOp module) {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::tt::ttcore::createTTCoreRegisterDevicePass());
    return pm.run(module);
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

  mlir::SmallVector<std::pair<mlir::Region *, mlir::Value>>
  getSingleOutputValuesFromLinalgOps(d2m::GenericOp genericOp) {
    mlir::SmallVector<std::pair<mlir::Region *, mlir::Value>> outputValues;
    genericOp->walk([&](mlir::linalg::GenericOp linalgOp) {
      if (linalgOp.getOutputs().size() == 1u) {
        outputValues.push_back(
            {linalgOp->getParentRegion(), linalgOp.getOutputs().front()});
      }
    });
    return outputValues;
  }

  static constexpr llvm::StringLiteral kModuleHeader = R"mlir(
module {
)mlir";

  static constexpr llvm::StringLiteral kModuleFooter = R"mlir(
}
)mlir";

  std::string wrapInModule(llvm::StringRef funcBody) {
    return (kModuleHeader + funcBody + kModuleFooter).str();
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
  std::string moduleText = wrapInModule(R"mlir(
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
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 1u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  auto it = regionInfo->perResult.find(outputValues.front().second);
  ASSERT_NE(it, regionInfo->perResult.end());
  ASSERT_EQ(regionInfo->perResult.size(), 1u);
  EXPECT_EQ(it->second.numTilesPerFlip, 2);
  EXPECT_EQ(it->second.numDstFlips, 2);
  EXPECT_EQ(regionInfo->numTilesPerResult, 4);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 2);
}

TEST_F(GenericOpAnalysisTest,
       CanAnalyzeGenericForDSTPackingWithoutBlockingLoop) {
  std::string moduleText = wrapInModule(R"mlir(
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
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in_m : memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
    }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 1u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  auto it = regionInfo->perResult.find(outputValues.front().second);
  ASSERT_NE(it, regionInfo->perResult.end());
  ASSERT_EQ(regionInfo->perResult.size(), 1u);
  EXPECT_EQ(it->second.numTilesPerFlip, 2);
  EXPECT_EQ(it->second.numDstFlips, 2);
  EXPECT_EQ(regionInfo->numTilesPerResult, 4);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 2);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDSTPackingSingleTileShardEarlyOut) {
  std::string moduleText = wrapInModule(R"mlir(
func.func @test(
    %in: memref<1x1x1x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x1x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in : memref<1x1x1x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x1x!ttcore.tile<32x32, f32>>) {
    ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>>,
              %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>>):
      %in_m = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>> -> memref<1x1x!ttcore.tile<32x32, f32>>
      %out_m = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>> -> memref<1x1x!ttcore.tile<32x32, f32>>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in_m : memref<1x1x!ttcore.tile<32x32, f32>>)
        outs(%out_m : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
    }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 1u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  auto it = regionInfo->perResult.find(outputValues.front().second);
  ASSERT_NE(it, regionInfo->perResult.end());
  ASSERT_EQ(regionInfo->perResult.size(), 1u);
  // Single-tile shards should bypass the general multi-flip packing path.
  EXPECT_EQ(it->second.numTilesPerFlip, 1);
  EXPECT_EQ(it->second.numDstFlips, 1);
  EXPECT_EQ(regionInfo->numTilesPerResult, 1);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 1);
}

TEST_F(GenericOpAnalysisTest,
       CanAnalyzeDSTPackingMultipleSingleTileShardsEarlyOut) {
  std::string moduleText = wrapInModule(R"mlir(
func.func @test(
    %in0: memref<1x1x1x!ttcore.tile<32x32, f32>>,
    %in1: memref<1x1x1x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x1x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in0, %in1 : memref<1x1x1x!ttcore.tile<32x32, f32>>, memref<1x1x1x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x1x!ttcore.tile<32x32, f32>>) {
  ^unified0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>>,
            %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>>,
            %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>>):
    %in0_m = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>> -> memref<1x1x!ttcore.tile<32x32, f32>>
    %in1_m = d2m.wait %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>> -> memref<1x1x!ttcore.tile<32x32, f32>>
    %out_m = d2m.reserve %cb2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>>> -> memref<1x1x!ttcore.tile<32x32, f32>>
    %out0_m = memref.cast %out_m : memref<1x1x!ttcore.tile<32x32, f32>> to memref<1x1x!ttcore.tile<32x32, f32>>
    %out1_m = memref.cast %out_m : memref<1x1x!ttcore.tile<32x32, f32>> to memref<1x1x!ttcore.tile<32x32, f32>>
    %out2_m = memref.cast %out_m : memref<1x1x!ttcore.tile<32x32, f32>> to memref<1x1x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1 step %c1 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x1x!ttcore.tile<32x32, f32>>)
        outs(%out0_m : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>)
        outs(%out1_m : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sum : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x1x!ttcore.tile<32x32, f32>>)
        outs(%out2_m : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %cst = arith.constant 2.0 : f32
        %mul = "d2m.tile_mul"(%arg0, %cst) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>
        linalg.yield %mul : !ttcore.tile<32x32, f32>
      }
    }
  }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 3u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  ASSERT_EQ(regionInfo->perResult.size(), 3u);
  for (const auto &[parentRegion, outputValue] : outputValues) {
    auto it = regionInfo->perResult.find(outputValue);
    ASSERT_NE(it, regionInfo->perResult.end());
    // Every result in the all-single-tile region should take the early-out.
    EXPECT_EQ(it->second.numTilesPerFlip, 1);
    EXPECT_EQ(it->second.numDstFlips, 1);
  }
  EXPECT_EQ(regionInfo->numTilesPerResult, 1);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 1);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingFPU) {
  std::string moduleText = wrapInModule(R"mlir(
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
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 1u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  auto it = regionInfo->perResult.find(outputValues.front().second);
  ASSERT_NE(it, regionInfo->perResult.end());
  ASSERT_EQ(regionInfo->perResult.size(), 1u);
  EXPECT_EQ(it->second.numTilesPerFlip, 4);
  EXPECT_EQ(it->second.numDstFlips, 2);
  EXPECT_EQ(regionInfo->numTilesPerResult, 8);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 1);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeDSTPackingForDistinctParentRegions) {
  std::string moduleText = wrapInModule(R"mlir(
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
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 2u);
  ASSERT_EQ(packing->size(), 2u);
  ASSERT_NE(outputValues[0].first, outputValues[1].first);
  for (const auto &[parentRegion, outputValue] : outputValues) {
    auto *regionInfo = packing->lookup(parentRegion);
    ASSERT_NE(regionInfo, nullptr);
    ASSERT_EQ(regionInfo->perResult.size(), 1u);
    auto it = regionInfo->perResult.find(outputValue);
    ASSERT_NE(it, regionInfo->perResult.end());
    EXPECT_EQ(it->second.numTilesPerFlip, 2);
    EXPECT_EQ(it->second.numDstFlips, 2);
    EXPECT_EQ(regionInfo->numTilesPerResult, 4);
    EXPECT_EQ(regionInfo->numOuterLoopIters, 2);
  }
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingManyMixedOps) {
  std::string moduleText = wrapInModule(R"mlir(
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
    %out0_m = memref.cast %out_m : memref<1x8x!ttcore.tile<32x32, f32>> to memref<1x8x!ttcore.tile<32x32, f32>>
    %out1_m = memref.cast %out_m : memref<1x8x!ttcore.tile<32x32, f32>> to memref<1x8x!ttcore.tile<32x32, f32>>
    %out2_m = memref.cast %out_m : memref<1x8x!ttcore.tile<32x32, f32>> to memref<1x8x!ttcore.tile<32x32, f32>>
    %out3_m = memref.cast %out_m : memref<1x8x!ttcore.tile<32x32, f32>> to memref<1x8x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out0_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x8x!ttcore.tile<32x32, f32>>, memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out1_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sum : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out2_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %cst = arith.constant 2.0 : f32
        %mul = "d2m.tile_mul"(%arg0, %cst) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>
        linalg.yield %mul : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x8x!ttcore.tile<32x32, f32>>, memref<1x8x!ttcore.tile<32x32, f32>>)
        outs(%out3_m : memref<1x8x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sub = "d2m.tile_sub"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sub : !ttcore.tile<32x32, f32>
      }
    }
  }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 4u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  ASSERT_EQ(regionInfo->perResult.size(), 4u);
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[0].second).numTilesPerFlip,
      2); // tile_abs => SFPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[1].second).numTilesPerFlip,
      4); // tile_add(tile,tile) => FPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[2].second).numTilesPerFlip,
      2); // tile_mul(tile,scalar) => SFPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[3].second).numTilesPerFlip,
      4); // tile_sub(tile,tile) => FPU fp32
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[0].second).numDstFlips,
            4);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[1].second).numDstFlips,
            2);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[2].second).numDstFlips,
            4);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[3].second).numDstFlips,
            2);
  EXPECT_EQ(regionInfo->numTilesPerResult, 8);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 1);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingPrimeShardShapes) {
  std::string moduleText = wrapInModule(R"mlir(
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
    %out0_m = memref.cast %out_m : memref<1x7x!ttcore.tile<32x32, f32>> to memref<1x7x!ttcore.tile<32x32, f32>>
    %out1_m = memref.cast %out_m : memref<1x7x!ttcore.tile<32x32, f32>> to memref<1x7x!ttcore.tile<32x32, f32>>
    %out2_m = memref.cast %out_m : memref<1x7x!ttcore.tile<32x32, f32>> to memref<1x7x!ttcore.tile<32x32, f32>>
    %out3_m = memref.cast %out_m : memref<1x7x!ttcore.tile<32x32, f32>> to memref<1x7x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c7 = arith.constant 7 : index
    scf.for %i = %c0 to %c7 step %c1 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out0_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x7x!ttcore.tile<32x32, f32>>, memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out1_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sum : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out2_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %cst = arith.constant 3.0 : f32
        %mul = "d2m.tile_mul"(%arg0, %cst) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>
        linalg.yield %mul : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x7x!ttcore.tile<32x32, f32>>, memref<1x7x!ttcore.tile<32x32, f32>>)
        outs(%out3_m : memref<1x7x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sub = "d2m.tile_sub"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sub : !ttcore.tile<32x32, f32>
      }
    }
  }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 4u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  ASSERT_EQ(regionInfo->perResult.size(), 4u);
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[0].second).numTilesPerFlip,
      1); // prime shard, SFPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[1].second).numTilesPerFlip,
      1); // prime shard, FPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[2].second).numTilesPerFlip,
      1); // prime shard, SFPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[3].second).numTilesPerFlip,
      1); // prime shard, FPU fp32
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[0].second).numDstFlips,
            7);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[1].second).numDstFlips,
            7);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[2].second).numDstFlips,
            7);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[3].second).numDstFlips,
            7);
  EXPECT_EQ(regionInfo->numTilesPerResult, 7);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 1);
}

TEST_F(GenericOpAnalysisTest,
       CanAnalyzeGenericForDSTPackingShardSize15MixedOps) {
  std::string moduleText = wrapInModule(R"mlir(
func.func @test(
    %in0: memref<1x1x15x!ttcore.tile<32x32, f32>>,
    %in1: memref<1x1x15x!ttcore.tile<32x32, f32>>,
    %out: memref<1x1x15x!ttcore.tile<32x32, f32>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in0, %in1 : memref<1x1x15x!ttcore.tile<32x32, f32>>, memref<1x1x15x!ttcore.tile<32x32, f32>>)
    outs(%out : memref<1x1x15x!ttcore.tile<32x32, f32>>) {
  ^unified0(%cb0: !d2m.cb<memref<1x15x!ttcore.tile<32x32, f32>>>,
            %cb1: !d2m.cb<memref<1x15x!ttcore.tile<32x32, f32>>>,
            %cb2: !d2m.cb<memref<1x15x!ttcore.tile<32x32, f32>>>):
    %in0_m = d2m.wait %cb0 : !d2m.cb<memref<1x15x!ttcore.tile<32x32, f32>>> -> memref<1x15x!ttcore.tile<32x32, f32>>
    %in1_m = d2m.wait %cb1 : !d2m.cb<memref<1x15x!ttcore.tile<32x32, f32>>> -> memref<1x15x!ttcore.tile<32x32, f32>>
    %out_m = d2m.reserve %cb2 : !d2m.cb<memref<1x15x!ttcore.tile<32x32, f32>>> -> memref<1x15x!ttcore.tile<32x32, f32>>
    %out0_m = memref.cast %out_m : memref<1x15x!ttcore.tile<32x32, f32>> to memref<1x15x!ttcore.tile<32x32, f32>>
    %out1_m = memref.cast %out_m : memref<1x15x!ttcore.tile<32x32, f32>> to memref<1x15x!ttcore.tile<32x32, f32>>
    %out2_m = memref.cast %out_m : memref<1x15x!ttcore.tile<32x32, f32>> to memref<1x15x!ttcore.tile<32x32, f32>>
    %out3_m = memref.cast %out_m : memref<1x15x!ttcore.tile<32x32, f32>> to memref<1x15x!ttcore.tile<32x32, f32>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    scf.for %i = %c0 to %c15 step %c1 {
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x15x!ttcore.tile<32x32, f32>>)
        outs(%out0_m : memref<1x15x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x15x!ttcore.tile<32x32, f32>>, memref<1x15x!ttcore.tile<32x32, f32>>)
        outs(%out1_m : memref<1x15x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sum : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m : memref<1x15x!ttcore.tile<32x32, f32>>)
        outs(%out2_m : memref<1x15x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
        %cst = arith.constant 2.0 : f32
        %mul = "d2m.tile_mul"(%arg0, %cst) : (!ttcore.tile<32x32, f32>, f32) -> !ttcore.tile<32x32, f32>
        linalg.yield %mul : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
        ins(%in0_m, %in1_m : memref<1x15x!ttcore.tile<32x32, f32>>, memref<1x15x!ttcore.tile<32x32, f32>>)
        outs(%out3_m : memref<1x15x!ttcore.tile<32x32, f32>>) {
      ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
        %sub = "d2m.tile_sub"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %sub : !ttcore.tile<32x32, f32>
      }
    }
  }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 4u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  ASSERT_EQ(regionInfo->perResult.size(), 4u);
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[0].second).numTilesPerFlip,
      1); // tile_abs => SFPU fp32, maxDst=2, only factor of 15 <= 2 is 1
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[1].second).numTilesPerFlip,
      3); // tile_add(tile,tile) => FPU fp32, maxDst=4, largest factor of
          // 15 <= 4 is 3
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[2].second).numTilesPerFlip,
      1); // tile_mul(tile,scalar) => SFPU fp32
  EXPECT_EQ(
      regionInfo->perResult.lookup(outputValues[3].second).numTilesPerFlip,
      3); // tile_sub(tile,tile) => FPU fp32
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[0].second).numDstFlips,
            15);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[1].second).numDstFlips,
            5);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[2].second).numDstFlips,
            15);
  EXPECT_EQ(regionInfo->perResult.lookup(outputValues[3].second).numDstFlips,
            5);
  EXPECT_EQ(regionInfo->numTilesPerResult, 15);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 1);
}

TEST_F(GenericOpAnalysisTest, CanAnalyzeGenericForDSTPackingFPUBf16Square8By8) {
  std::string moduleText = wrapInModule(R"mlir(
func.func @test(
    %in0: memref<1x1x8x8x!ttcore.tile<32x32, bf16>>,
    %in1: memref<1x1x8x8x!ttcore.tile<32x32, bf16>>,
    %out: memref<1x1x8x8x!ttcore.tile<32x32, bf16>>) {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
    ins(%in0, %in1 : memref<1x1x8x8x!ttcore.tile<32x32, bf16>>, memref<1x1x8x8x!ttcore.tile<32x32, bf16>>)
    outs(%out : memref<1x1x8x8x!ttcore.tile<32x32, bf16>>) {
    ^unified0(%cb0: !d2m.cb<memref<8x8x!ttcore.tile<32x32, bf16>>>,
              %cb1: !d2m.cb<memref<8x8x!ttcore.tile<32x32, bf16>>>,
              %cb2: !d2m.cb<memref<8x8x!ttcore.tile<32x32, bf16>>>):
      %in0_m = d2m.wait %cb0 : !d2m.cb<memref<8x8x!ttcore.tile<32x32, bf16>>> -> memref<8x8x!ttcore.tile<32x32, bf16>>
      %in1_m = d2m.wait %cb1 : !d2m.cb<memref<8x8x!ttcore.tile<32x32, bf16>>> -> memref<8x8x!ttcore.tile<32x32, bf16>>
      %out_m = d2m.reserve %cb2 : !d2m.cb<memref<8x8x!ttcore.tile<32x32, bf16>>> -> memref<8x8x!ttcore.tile<32x32, bf16>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c64 step %c1 {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
          ins(%in0_m, %in1_m : memref<8x8x!ttcore.tile<32x32, bf16>>, memref<8x8x!ttcore.tile<32x32, bf16>>)
          outs(%out_m : memref<8x8x!ttcore.tile<32x32, bf16>>) {
        ^bb0(%arg0: !ttcore.tile<32x32, bf16>, %arg1: !ttcore.tile<32x32, bf16>, %arg2: !ttcore.tile<32x32, bf16>):
          %sum = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
          linalg.yield %sum : !ttcore.tile<32x32, bf16>
        }
      }
    }
  return
}
)mlir");

  auto module = parseModule(moduleText);
  ASSERT_TRUE(module);
  d2m::GenericOp generic = getSingleGenericOp(*module);
  ASSERT_TRUE(generic);
  markAllForLoopsAsBlocking(generic);

  d2m::utils::DstRegisterAnalysis analysis(generic);
  const auto *packing = analysis.lookup(generic);
  ASSERT_NE(packing, nullptr);
  auto outputValues = getSingleOutputValuesFromLinalgOps(generic);
  ASSERT_EQ(outputValues.size(), 1u);
  ASSERT_EQ(packing->size(), 1u);
  auto *regionInfo = packing->lookup(outputValues.front().first);
  ASSERT_NE(regionInfo, nullptr);
  auto it = regionInfo->perResult.find(outputValues.front().second);
  ASSERT_NE(it, regionInfo->perResult.end());
  ASSERT_EQ(regionInfo->perResult.size(), 1u);
  EXPECT_EQ(it->second.numTilesPerFlip, 8);
  EXPECT_EQ(it->second.numDstFlips, 2);
  EXPECT_EQ(regionInfo->numTilesPerResult, 16);
  EXPECT_EQ(regionInfo->numOuterLoopIters, 4);
}

} // namespace mlir::tt::d2m
