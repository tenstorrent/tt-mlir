// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "gtest/gtest.h"

namespace mlir::tt::d2m {

class BlockFactorAnalysisTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;

  void SetUp() override {
    // Initialize context and load necessary dialects.
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttir::TTIRDialect>();
    context.loadDialect<mlir::tt::d2m::D2MDialect>();

    // Create a simple module with a function containing a single block.
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    auto funcType = builder.getFunctionType({}, {});
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                              "test_func", funcType);

    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
  }

  void TearDown() override {}
};

// Create a generic op with the given grid and dimensions.
GenericOp createGenericOp(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                          llvm::SmallVector<int64_t> grid,
                          llvm::SmallVector<int64_t> dims) {

  auto tileType = mlir::tt::ttcore::TileType::get(
      &context, {32, 32}, mlir::tt::ttcore::DataType::Float32);

  auto l1MemorySpace = mlir::tt::ttcore::MemorySpaceAttr::get(
      &context, mlir::tt::ttcore::MemorySpace::DeviceL1);

  assert(grid.size() == 2);
  assert(dims.size() == grid.size());

  auto shardShape = llvm::SmallVector<int64_t>(dims.size(), 0);
  for (size_t i = 0; i < dims.size(); i++) {
    shardShape[i] = dims[i] / grid[i];
  }
  auto memrefShape = llvm::SmallVector<int64_t>(grid);
  memrefShape.insert(memrefShape.end(), shardShape.begin(), shardShape.end());

  auto tileSize = tileType.getSizeBytes();
  auto shardStrides = llvm::SmallVector<int64_t>(shardShape.size(), 0);
  int64_t strideAccum = tileSize;
  for (size_t i = 0; i < shardShape.size(); i++) {
    shardStrides[i] = strideAccum;
    strideAccum *= shardShape[i];
  }

  auto shardLayout1 = mlir::tt::ttcore::ShardLayoutAttr::get(
      &context, shardStrides, /*buffered=*/1);
  auto shardLayout2 = mlir::tt::ttcore::ShardLayoutAttr::get(
      &context, shardStrides, /*buffered=*/1);
  auto shardLayout3 = mlir::tt::ttcore::ShardLayoutAttr::get(
      &context, shardStrides, /*buffered=*/1);

  auto memrefType1 =
      mlir::MemRefType::get(memrefShape, tileType, shardLayout1, l1MemorySpace);
  auto memrefType2 =
      mlir::MemRefType::get(memrefShape, tileType, shardLayout2, l1MemorySpace);
  auto memrefType3 =
      mlir::MemRefType::get(memrefShape, tileType, shardLayout3, l1MemorySpace);

  auto identityMap2D = mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  auto indexingMaps = builder.getAffineMapArrayAttr(
      {identityMap2D, identityMap2D, identityMap2D});

  auto parallelType = mlir::tt::ttcore::IteratorTypeAttr::get(
      &context, mlir::tt::ttcore::IteratorType::Parallel);
  auto iteratorTypes = builder.getArrayAttr({parallelType, parallelType});

  // Use memref alloc to create concrete memref values from types.
  Value input1 =
      builder.create<ttir::EmptyOp>(builder.getUnknownLoc(), memrefType1)
          ->getResult(0);
  Value input2 =
      builder.create<ttir::EmptyOp>(builder.getUnknownLoc(), memrefType2)
          ->getResult(0);
  Value output =
      builder.create<ttir::EmptyOp>(builder.getUnknownLoc(), memrefType3)
          ->getResult(0);

  SmallVector<Value> inputs = {input1, input2};
  SmallVector<Value> outputs = {output};

  // Create the GenericOp.
  auto genericOp = builder.create<mlir::tt::d2m::GenericOp>(
      builder.getUnknownLoc(), inputs, outputs, indexingMaps, iteratorTypes);
  return genericOp;
}

using Constraints = mlir::tt::d2m::BlockFactorAnalysisConstraints;
using Strategy = Constraints::BufferStrategy;

TEST_F(BlockFactorAnalysisTest, CanCreateGenericOpsWithDifferentGrids) {

  auto genericOp1 = createGenericOp(builder, context, {1, 1}, {2, 4});
  EXPECT_TRUE(genericOp1);
  EXPECT_EQ(genericOp1.getInputs().size(), 2u);
  EXPECT_EQ(genericOp1.getOutputs().size(), 1u);
  auto grid1 = genericOp1.getGrid();
  EXPECT_EQ(grid1.getShape()[0], 1);
  EXPECT_EQ(grid1.getShape()[1], 1);

  auto genericOp2 = createGenericOp(builder, context, {2, 2}, {4, 8});
  EXPECT_TRUE(genericOp2);
  auto grid2 = genericOp2.getGrid();
  EXPECT_EQ(grid2.getShape()[0], 2);
  EXPECT_EQ(grid2.getShape()[1], 2);

  auto genericOp3 = createGenericOp(builder, context, {1, 4}, {2, 8});
  EXPECT_TRUE(genericOp3);
  auto grid3 = genericOp3.getGrid();
  EXPECT_EQ(grid3.getShape()[0], 1);
  EXPECT_EQ(grid3.getShape()[1], 4);
}

// Value-parameterized tests for different sharding configurations
struct BlockFactorAnalysisTestParams {
  llvm::SmallVector<int64_t> grid;
  llvm::SmallVector<int64_t> dims;
  llvm::SmallVector<int64_t> expectedBufferShape;
  std::string testName;
};

class BlockFactorAnalysisParamTest : public BlockFactorAnalysisTest,
                                     public ::testing::WithParamInterface<BlockFactorAnalysisTestParams> {};

TEST_P(BlockFactorAnalysisParamTest, CanPerformBlockFactorAnalysis) {
  auto params = GetParam();
  
  auto genericOp = createGenericOp(builder, context, params.grid, params.dims);

  BlockFactorAnalysis analysis;
  auto bufferConfigs = analysis.analyzeGenericOp(genericOp);

  ASSERT_FALSE(bufferConfigs.empty());
  const auto &config = bufferConfigs[0];
  EXPECT_EQ(config.operandBufferSettings.size(), 3u);

  for (const auto &setting : config.operandBufferSettings) {
    EXPECT_FALSE(setting.bufferShape.empty());
    EXPECT_GT(setting.numBuffers, 0u);
  }
  EXPECT_EQ(config.operandBufferSettings[0].bufferShape, params.expectedBufferShape);
}

INSTANTIATE_TEST_SUITE_P(
    BlockFactorAnalysisTests, BlockFactorAnalysisParamTest,
    ::testing::Values(
        BlockFactorAnalysisTestParams{{1, 1}, {2, 4}, {2, 4}, "1x1Sharded"},
        BlockFactorAnalysisTestParams{{2, 2}, {4, 8}, {2, 4}, "2x2Sharded"},
        BlockFactorAnalysisTestParams{{1, 4}, {2, 8}, {2, 2}, "1x4Sharded"}
    ));

TEST_F(BlockFactorAnalysisTest, CanAnalyzeBufferingStrategies) {

  auto genericOp = createGenericOp(builder, context, {1, 1}, {2, 4});

  BlockFactorAnalysisConstraints singleBufferConstraints{
      Constraints::BufferStrategy::SingleBuffered};
  mlir::tt::d2m::BlockFactorAnalysis analysisSingle(singleBufferConstraints);
  auto bufferConfigsSingle = analysisSingle.analyzeGenericOp(genericOp);

  ASSERT_FALSE(bufferConfigsSingle.empty());
  for (const auto &setting : bufferConfigsSingle[0].operandBufferSettings) {
    EXPECT_EQ(setting.numBuffers, 1u);
  }

  Constraints doubleBufferConstraints{
      Constraints::BufferStrategy::DoubleBuffered};
  mlir::tt::d2m::BlockFactorAnalysis analysisDouble(doubleBufferConstraints);
  auto bufferConfigsDouble = analysisDouble.analyzeGenericOp(genericOp);

  ASSERT_FALSE(bufferConfigsDouble.empty());
  const auto &configDouble = bufferConfigsDouble[0];
  EXPECT_EQ(configDouble.operandBufferSettings.size(), 3u);
  for (const auto &setting : configDouble.operandBufferSettings) {
    EXPECT_EQ(setting.numBuffers, 2u);
  }
}

} // namespace mlir::tt::d2m
