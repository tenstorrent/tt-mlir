// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/LoopSemantics.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::tt;

class LoopSemanticsTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> ctx;

  void SetUp() override {
    ctx = std::make_unique<MLIRContext>();
    ctx->loadDialect<func::FuncDialect, d2m::D2MDialect, ttcore::TTCoreDialect,
                     memref::MemRefDialect>();
  }

  // Helper to create a matmul-like GenericOp for testing.
  d2m::GenericOp createMatmulGenericOp(OpBuilder &builder) {
    auto loc = builder.getUnknownLoc();

    // Create module and function
    auto module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    auto funcType = builder.getFunctionType({}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, "test_func", funcType);
    funcOp.addEntryBlock();
    builder.setInsertionPointToEnd(&funcOp.getBody().front());

    // Create memref types for A[i,k], B[k,j], C[i,j].
    auto f32Type = builder.getF32Type();
    auto memrefAType = MemRefType::get({3, 3}, f32Type);
    auto memrefBType = MemRefType::get({3, 2}, f32Type);
    auto memrefCType = MemRefType::get({3, 2}, f32Type);

    // Create dummy memref values.
    auto allocA = builder.create<memref::AllocOp>(loc, memrefAType);
    auto allocB = builder.create<memref::AllocOp>(loc, memrefBType);
    auto allocC = builder.create<memref::AllocOp>(loc, memrefCType);

    // Create indexing maps for matmul:
    // A[i, k] -> (d0, d1, d2) -> (d0, d2)
    // B[k, j] -> (d0, d1, d2) -> (d2, d1)
    // C[i, j] -> (d0, d1, d2) -> (d0, d1)
    auto mapA = AffineMap::get(
        3, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(2)},
        builder.getContext());
    auto mapB = AffineMap::get(
        3, 0, {builder.getAffineDimExpr(2), builder.getAffineDimExpr(1)},
        builder.getContext());
    auto mapC = AffineMap::get(
        3, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
        builder.getContext());

    ArrayAttr indexingMaps = builder.getAffineMapArrayAttr({mapA, mapB, mapC});

    // Create iterator types: [parallel, parallel, reduction].
    auto parallelAttr = builder.getAttr<ttcore::IteratorTypeAttr>(
        ttcore::IteratorType::Parallel);
    auto reductionAttr = builder.getAttr<ttcore::IteratorTypeAttr>(
        ttcore::IteratorType::Reduction);
    ArrayAttr iteratorTypes =
        builder.getArrayAttr({parallelAttr, parallelAttr, reductionAttr});

    // Create grid and block_factors attributes.
    SmallVector<int64_t> gridShape = {1, 1};
    auto gridAttr = ttcore::GridAttr::get(builder.getContext(), gridShape);
    SmallVector<int64_t> blockFactors = {1, 1, 1};

    // Create GenericOp with explicit attributes to avoid grid shape inference
    SmallVector<Value> inputValues = {allocA.getResult(), allocB.getResult()};
    SmallVector<Value> outputValues = {allocC.getResult()};
    ValueRange inputs(inputValues);
    ValueRange outputs(outputValues);

    auto genericOp = builder.create<d2m::GenericOp>(
        loc, inputs, outputs, indexingMaps, iteratorTypes,
        d2m::ThreadType::Compute, gridAttr, blockFactors);

    return genericOp;
  }
};

TEST_F(LoopSemanticsTest, AnalyzeMatmulDimensions) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  const auto &dimInfo = analyzer.getDimensionInfo();

  // Matmul has 3 dimensions: [parallel, parallel, reduction].
  EXPECT_EQ(dimInfo.numDimensions, 3u);
  EXPECT_EQ(dimInfo.iteratorTypes.size(), 3u);

  // Check iterator types.
  EXPECT_EQ(dimInfo.iteratorTypes[0], ttcore::IteratorType::Parallel);
  EXPECT_EQ(dimInfo.iteratorTypes[1], ttcore::IteratorType::Parallel);
  EXPECT_EQ(dimInfo.iteratorTypes[2], ttcore::IteratorType::Reduction);

  // Check dimension classification
  EXPECT_TRUE(dimInfo.isParallel(0));
  EXPECT_TRUE(dimInfo.isParallel(1));
  EXPECT_FALSE(dimInfo.isParallel(2));

  EXPECT_FALSE(dimInfo.isReduction(0));
  EXPECT_FALSE(dimInfo.isReduction(1));
  EXPECT_TRUE(dimInfo.isReduction(2));

  // Check sets.
  EXPECT_TRUE(dimInfo.parallelDims.contains(0));
  EXPECT_TRUE(dimInfo.parallelDims.contains(1));
  EXPECT_FALSE(dimInfo.parallelDims.contains(2));

  EXPECT_FALSE(dimInfo.reductionDims.contains(0));
  EXPECT_FALSE(dimInfo.reductionDims.contains(1));
  EXPECT_TRUE(dimInfo.reductionDims.contains(2));
}

TEST_F(LoopSemanticsTest, GetOutputAccessInfo) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  // Get access info for output C (operand index 2, output index 0).
  auto accessInfo = analyzer.getOperandAccessInfo(2);

  // Output C is at operand index 2
  EXPECT_EQ(accessInfo.operandIndex, 2u);
  EXPECT_TRUE(accessInfo.isOutput);

  // Output C[i,j] participates in dimensions 0 (i) and 1 (j).
  EXPECT_EQ(accessInfo.participatingDims.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(accessInfo.participatingDims, 0));
  EXPECT_TRUE(llvm::is_contained(accessInfo.participatingDims, 1));

  // Output C doesn't participate in dimension 2 (k - reduction)
  EXPECT_EQ(accessInfo.nonParticipatingDims.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(accessInfo.nonParticipatingDims, 2));

  // Test helper methods.
  EXPECT_TRUE(accessInfo.usesDimension(0));
  EXPECT_TRUE(accessInfo.usesDimension(1));
  EXPECT_FALSE(accessInfo.usesDimension(2));

  EXPECT_FALSE(accessInfo.doesNotUseDimension(0));
  EXPECT_FALSE(accessInfo.doesNotUseDimension(1));
  EXPECT_TRUE(accessInfo.doesNotUseDimension(2));
}

TEST_F(LoopSemanticsTest, GetInputAccessInfo) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  // Get access info for input A (operand index 0).
  // A[i, k] -> participates in dims 0 (i) and 2 (k).
  auto accessInfoA = analyzer.getOperandAccessInfo(0);

  EXPECT_EQ(accessInfoA.operandIndex, 0u);
  EXPECT_FALSE(accessInfoA.isOutput);
  EXPECT_EQ(accessInfoA.participatingDims.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(accessInfoA.participatingDims, 0));
  EXPECT_TRUE(llvm::is_contained(accessInfoA.participatingDims, 2));
  EXPECT_EQ(accessInfoA.nonParticipatingDims.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(accessInfoA.nonParticipatingDims, 1));

  // Get access info for input B (operand index 1).
  // B[k, j] -> participates in dims 1 (j) and 2 (k).
  auto accessInfoB = analyzer.getOperandAccessInfo(1);

  EXPECT_EQ(accessInfoB.operandIndex, 1u);
  EXPECT_FALSE(accessInfoB.isOutput);
  EXPECT_EQ(accessInfoB.participatingDims.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(accessInfoB.participatingDims, 1));
  EXPECT_TRUE(llvm::is_contained(accessInfoB.participatingDims, 2));
  EXPECT_EQ(accessInfoB.nonParticipatingDims.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(accessInfoB.nonParticipatingDims, 0));
}

TEST_F(LoopSemanticsTest, GetPrologueEpilogueDims) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  // For matmul output C[i,j], prologue/epilogue should only use dims {0, 1},
  // NOT dimension 2 (reduction).
  auto prologueDims = analyzer.getPrologueEpilogueDims(0);

  EXPECT_EQ(prologueDims.size(), 2u);
  EXPECT_TRUE(llvm::is_contained(prologueDims, 0u));
  EXPECT_TRUE(llvm::is_contained(prologueDims, 1u));
  EXPECT_FALSE(llvm::is_contained(prologueDims, 2u));
}

TEST_F(LoopSemanticsTest, GetGuardDims) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  // For matmul output C (operand index 2), guard dims should be {2}
  // (the reduction dimension that C doesn't participate in).
  auto guardDims = analyzer.getGuardDims(2);

  EXPECT_EQ(guardDims.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(guardDims, 2u));

  // For input A (operand index 0), guard dim should be {1}
  // (A doesn't use dimension j.)
  auto guardDimsA = analyzer.getGuardDims(0);
  EXPECT_EQ(guardDimsA.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(guardDimsA, 1u));

  // For input B (operand index 1), guard dim should be {0}.
  // (B doesn't use dimension i.)
  auto guardDimsB = analyzer.getGuardDims(1);
  EXPECT_EQ(guardDimsB.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(guardDimsB, 0u));
}

TEST_F(LoopSemanticsTest, GetOutputAccessInfos) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  auto outputInfos = analyzer.getOutputAccessInfos();

  // Matmul has one output.
  EXPECT_EQ(outputInfos.size(), 1u);

  // Check the output info
  EXPECT_TRUE(outputInfos[0].isOutput);
  EXPECT_EQ(outputInfos[0].participatingDims.size(), 2u);
  EXPECT_EQ(outputInfos[0].nonParticipatingDims.size(), 1u);
}

TEST_F(LoopSemanticsTest, GetGenericOp) {
  OpBuilder builder(ctx.get());
  auto genericOp = createMatmulGenericOp(builder);

  d2m::utils::LoopSemanticsAnalyzer analyzer(genericOp);

  // Verify we can get the GenericOp back.
  auto retrievedOp = analyzer.getGenericOp();
  EXPECT_EQ(retrievedOp, genericOp);
}
