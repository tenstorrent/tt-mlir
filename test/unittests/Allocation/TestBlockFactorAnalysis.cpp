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
    // Initialize context and load necessary dialects
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttir::TTIRDialect>();
    context.loadDialect<mlir::tt::d2m::D2MDialect>();

    // Create a simple module with a function
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    // Note: Device registration may be needed for some layouts but not for
    // basic testing

    // Create a function
    auto funcType = builder.getFunctionType({}, {});
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                              "test_func", funcType);

    // Create a basic block in the function
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
  }

  void TearDown() override {}
};

GenericOp createGenericOp(mlir::OpBuilder &builder, mlir::MLIRContext &context,
                          llvm::SmallVector<int64_t> grid,
                          llvm::SmallVector<int64_t> dims) {

  // Create tile type with default 32x32 shape and f32 data type
  auto tileType = mlir::tt::ttcore::TileType::get(
      &context, {32, 32}, mlir::tt::ttcore::DataType::Float32);

  // Create memory space attribute for L1 memory
  auto l1MemorySpace = mlir::tt::ttcore::MemorySpaceAttr::get(
      &context, mlir::tt::ttcore::MemorySpace::DeviceL1);

  assert(grid.size() == 2);
  assert(dims.size() == grid.size());

  // divide dims by grid to get the shard shape
  auto shardShape = llvm::SmallVector<int64_t>(dims.size(), 0);
  for (size_t i = 0; i < dims.size(); i++) {
    shardShape[i] = dims[i] / grid[i];
  }
  // full memref shape is grid concat shard shape
  auto memrefShape = llvm::SmallVector<int64_t>(grid);
  memrefShape.insert(memrefShape.end(), shardShape.begin(), shardShape.end());

  auto tileSize = tileType.getSizeBytes();
  auto shardStrides = llvm::SmallVector<int64_t>(shardShape.size(), 0);
  int64_t stride_accum = tileSize;
  for (size_t i = 0; i < shardShape.size(); i++) {
    shardStrides[i] = stride_accum;
    stride_accum *= shardShape[i];
  }

  // Create shard layout attributes for the memref types
  auto shardLayout1 = mlir::tt::ttcore::ShardLayoutAttr::get(
      &context, shardStrides, /*buffered=*/1);
  auto shardLayout2 = mlir::tt::ttcore::ShardLayoutAttr::get(
      &context, shardStrides, /*buffered=*/1);
  auto shardLayout3 = mlir::tt::ttcore::ShardLayoutAttr::get(
      &context, shardStrides, /*buffered=*/1);

  // Create MemRefType for the three operands
  auto memrefType1 =
      mlir::MemRefType::get(memrefShape, tileType, shardLayout1, l1MemorySpace);
  auto memrefType2 =
      mlir::MemRefType::get(memrefShape, tileType, shardLayout2, l1MemorySpace);
  auto memrefType3 =
      mlir::MemRefType::get(memrefShape, tileType, shardLayout3, l1MemorySpace);

  // Create affine maps for indexing (identity maps for simplicity)
  auto identityMap2D = mlir::AffineMap::getMultiDimIdentityMap(2, &context);
  auto indexingMaps = builder.getAffineMapArrayAttr(
      {identityMap2D, identityMap2D, identityMap2D});

  // Create iterator types (parallel for all dimensions)
  auto parallelType = mlir::tt::ttcore::IteratorTypeAttr::get(
      &context, mlir::tt::ttcore::IteratorType::Parallel);
  auto iteratorTypes = builder.getArrayAttr({parallelType, parallelType});

  // use memref alloc to create concrete memref values from types
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

  // Create the GenericOp
  auto genericOp = builder.create<mlir::tt::d2m::GenericOp>(
      builder.getUnknownLoc(), inputs, outputs, indexingMaps, iteratorTypes);
  return genericOp;
}

using Constraints = mlir::tt::d2m::BlockFactorAnalysisConstraints;
using Strategy = Constraints::BufferStrategy;

TEST_F(BlockFactorAnalysisTest, CanCreateGenericOpsWithDifferentGrids) {
  // Test creating GenericOps with different grid configurations

  // Test 1x1 grid with 2x4 dimensions
  auto genericOp1 = createGenericOp(builder, context, {1, 1}, {2, 4});
  EXPECT_TRUE(genericOp1);
  EXPECT_EQ(genericOp1.getInputs().size(), 2u);
  EXPECT_EQ(genericOp1.getOutputs().size(), 1u);
  auto grid1 = genericOp1.getGrid();
  EXPECT_EQ(grid1.getShape()[0], 1);
  EXPECT_EQ(grid1.getShape()[1], 1);

  // Test 2x2 grid with 4x8 dimensions
  auto genericOp2 = createGenericOp(builder, context, {2, 2}, {4, 8});
  EXPECT_TRUE(genericOp2);
  auto grid2 = genericOp2.getGrid();
  EXPECT_EQ(grid2.getShape()[0], 2);
  EXPECT_EQ(grid2.getShape()[1], 2);

  // Test 1x4 grid with 2x8 dimensions
  auto genericOp3 = createGenericOp(builder, context, {1, 4}, {2, 8});
  EXPECT_TRUE(genericOp3);
  auto grid3 = genericOp3.getGrid();
  EXPECT_EQ(grid3.getShape()[0], 1);
  EXPECT_EQ(grid3.getShape()[1], 4);
}

TEST_F(BlockFactorAnalysisTest,
       CanPerformBlockFactorAnalysis1x1ShardedGenericOp) {

  // Test Case 1: Small single-core operation (1x1 grid, 2x4 dims)
  auto genericOp1 = createGenericOp(builder, context, {1, 1}, {2, 4});

  BlockFactorAnalysis analysis1;
  auto bufferConfigs1 = analysis1.analyzeGenericOp(genericOp1);

  EXPECT_FALSE(bufferConfigs1.empty());
  if (!bufferConfigs1.empty()) {
    const auto &config1 = bufferConfigs1[0];
    EXPECT_EQ(config1.operand_buffer_settings.size(),
              3u); // 2 inputs + 1 output

    // Verify all operands have buffer settings
    for (const auto &setting : config1.operand_buffer_settings) {
      EXPECT_FALSE(setting.buffer_shape.empty());
      EXPECT_GT(setting.num_buffers, 0u);
    }
    EXPECT_EQ(config1.operand_buffer_settings[0].buffer_shape,
              (llvm::SmallVector<int64_t>{2, 4}));
  }
}

TEST_F(BlockFactorAnalysisTest,
       CanPerformBlockFactorAnalysis2x2ShardedGenericOp) {

  auto genericOp2 = createGenericOp(builder, context, {2, 2}, {4, 8});

  BlockFactorAnalysis analysis2;
  auto bufferConfigs2 = analysis2.analyzeGenericOp(genericOp2);

  EXPECT_FALSE(bufferConfigs2.empty());
  if (!bufferConfigs2.empty()) {
    const auto &config2 = bufferConfigs2[0];
    EXPECT_EQ(config2.operand_buffer_settings.size(), 3u);

    // Verify buffer configurations are reasonable
    for (const auto &setting : config2.operand_buffer_settings) {
      EXPECT_FALSE(setting.buffer_shape.empty());
      EXPECT_GT(setting.num_buffers, 0u);
    }
    EXPECT_EQ(config2.operand_buffer_settings[0].buffer_shape,
              (llvm::SmallVector<int64_t>{2, 4}));
  }
}

TEST_F(BlockFactorAnalysisTest,
       CanPerformBlockFactorAnalysis1x4ShardedGenericOp) {

  auto genericOp3 = createGenericOp(builder, context, {1, 4}, {2, 8});

  BlockFactorAnalysis analysis3;
  auto bufferConfigs3 = analysis3.analyzeGenericOp(genericOp3);

  EXPECT_FALSE(bufferConfigs3.empty());
  if (!bufferConfigs3.empty()) {
    const auto &config3 = bufferConfigs3[0];
    EXPECT_EQ(config3.operand_buffer_settings.size(), 3u);

    // Verify buffer configurations
    for (const auto &setting : config3.operand_buffer_settings) {
      EXPECT_FALSE(setting.buffer_shape.empty());
      EXPECT_GT(setting.num_buffers, 0u);
    }
    EXPECT_EQ(config3.operand_buffer_settings[0].buffer_shape,
              (llvm::SmallVector<int64_t>{2, 2}));
  }
}

TEST_F(BlockFactorAnalysisTest, CanAnalyzeBufferingStrategies) {

  auto genericOp = createGenericOp(builder, context, {1, 1}, {2, 4});

  BlockFactorAnalysisConstraints singleBufferConstraints{
      Constraints::BufferStrategy::SingleBuffered};
  mlir::tt::d2m::BlockFactorAnalysis analysisSingle(singleBufferConstraints);
  auto bufferConfigsSingle = analysisSingle.analyzeGenericOp(genericOp);

  EXPECT_FALSE(bufferConfigsSingle.empty());
  for (const auto &setting : bufferConfigsSingle[0].operand_buffer_settings) {
    EXPECT_EQ(setting.num_buffers, 1u);
  }

  Constraints doubleBufferConstraints{
      Constraints::BufferStrategy::DoubleBuffered};
  mlir::tt::d2m::BlockFactorAnalysis analysisDouble(doubleBufferConstraints);
  auto bufferConfigsDouble = analysisDouble.analyzeGenericOp(genericOp);

  EXPECT_FALSE(bufferConfigsDouble.empty());
  const auto &configDouble = bufferConfigsDouble[0];
  EXPECT_EQ(configDouble.operand_buffer_settings.size(), 3u);
  for (const auto &setting : configDouble.operand_buffer_settings) {
    EXPECT_EQ(setting.num_buffers, 2u);
  }
}

} // namespace mlir::tt::d2m
