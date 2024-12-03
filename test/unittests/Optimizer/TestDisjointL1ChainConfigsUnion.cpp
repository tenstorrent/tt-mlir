// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/Analysis/DisjointL1ChainConfigsUnion.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"

using namespace mlir::tt::ttnn;

constexpr int TensorDimX = 128;
constexpr int TensorDimY = 128;

class DisjoinL1ChainConfigsUnionBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;

  DisjoinL1ChainConfigsUnion disjointL1ChainConfigsUnion;

  void SetUp() override {
    context.loadDialect<TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    createFuncOp();

    disjointL1ChainConfigsUnion = DisjoinL1ChainConfigsUnion();
  }

  llvm::SmallVector<int64_t, 2> getTensorShape() {
    return {TensorDimX, TensorDimY};
  }

  mlir::RankedTensorType getTensorRankedType() {
    return mlir::RankedTensorType::get(getTensorShape(), builder.getF32Type());
  }

  mlir::Value createEmptyTensor() {
    ShapeAttr shapeAttr = ShapeAttr::get(&context, getTensorShape());
    return builder.create<EmptyOp>(builder.getUnknownLoc(),
                                   getTensorRankedType(), nullptr, shapeAttr,
                                   nullptr, nullptr, nullptr);
  }

  mlir::func::FuncOp createFuncOp() {
    mlir::SmallVector<mlir::Type> input;
    input.push_back(getTensorRankedType());

    mlir::SmallVector<mlir::Type> output;
    output.push_back(getTensorRankedType());

    auto funcType = builder.getType<mlir::FunctionType>(
        mlir::TypeRange(input), mlir::TypeRange(output));
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test",
                                              funcType);

    mlir::Block *block = func.addEntryBlock();
    block->addArgument(getTensorRankedType(), builder.getUnknownLoc());
    block->addArgument(getTensorRankedType(), builder.getUnknownLoc());

    builder.setInsertionPointToStart(block);

    return func;
  }

  mlir::Operation *createOp() {
    mlir::Value dest = createEmptyTensor();
    mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
    mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
    return builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  }

  void TearDown() override {}
};

TEST_F(DisjoinL1ChainConfigsUnionBase, TestInsertL1ChainConfig) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfigA, l1ChainConfigB, l1ChainConfigC;
  l1ChainConfigA.addOpL1MemSpec(memSpecOpA);
  l1ChainConfigB.addOpL1MemSpec(memSpecOpB);
  l1ChainConfigC.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigA);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigB);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfL1Chains(), 3);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestInsertOpInL1ChainConfig1) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  disjointL1ChainConfigsUnion.insertOpL1MemSpec(memSpecOpA, nullptr);
  disjointL1ChainConfigsUnion.insertOpL1MemSpec(memSpecOpB, nullptr);
  disjointL1ChainConfigsUnion.insertOpL1MemSpec(memSpecOpC, nullptr);

  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfL1Chains(), 3);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opB);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opC);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestInsertOpInL1ChainConfig2) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  disjointL1ChainConfigsUnion.insertOpL1MemSpec(memSpecOpA, nullptr);
  disjointL1ChainConfigsUnion.insertOpL1MemSpec(memSpecOpB, opA);
  disjointL1ChainConfigsUnion.insertOpL1MemSpec(memSpecOpC, opB);

  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfL1Chains(), 1);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opA);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestFindRepresentativeOp) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfigA, l1ChainConfigB, l1ChainConfigC;
  l1ChainConfigA.addOpL1MemSpec(memSpecOpA);
  l1ChainConfigB.addOpL1MemSpec(memSpecOpB);
  l1ChainConfigC.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigA);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigB);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opB);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opC);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestFindRepresentativeOpRecursive1) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfig;
  l1ChainConfig.addOpL1MemSpec(memSpecOpA);
  l1ChainConfig.addOpL1MemSpec(memSpecOpB);
  l1ChainConfig.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfig);

  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opA);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestFindRepresentativeOpRecursive2) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfigA, l1ChainConfigB, l1ChainConfigC;
  l1ChainConfigA.addOpL1MemSpec(memSpecOpA);
  l1ChainConfigB.addOpL1MemSpec(memSpecOpB);
  l1ChainConfigC.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigA);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigB);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opB);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opC);

  disjointL1ChainConfigsUnion.mergeL1ChainConfigs(opA, opB);

  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opC);

  disjointL1ChainConfigsUnion.mergeL1ChainConfigs(opB, opC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opA);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestMergeL1ChainConfigs) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfigA, l1ChainConfigB, l1ChainConfigC;
  l1ChainConfigA.addOpL1MemSpec(memSpecOpA);
  l1ChainConfigB.addOpL1MemSpec(memSpecOpB);
  l1ChainConfigC.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigA);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigB);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfigC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfL1Chains(), 3);
  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfOpsInChain(opA), 1);
  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfOpsInChain(opB), 1);
  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfOpsInChain(opC), 1);

  mlir::Operation *opD;
  opD = disjointL1ChainConfigsUnion.mergeL1ChainConfigs(opB, opC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfL1Chains(), 2);
  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfOpsInChain(opA), 1);
  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfOpsInChain(opD), 2);

  mlir::Operation *opE;
  opE = disjointL1ChainConfigsUnion.mergeL1ChainConfigs(opA, opD);

  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfL1Chains(), 1);
  ASSERT_EQ(disjointL1ChainConfigsUnion.getNumberOfOpsInChain(opE), 3);
}

// Smaller chain should always be merged into the larger chain
TEST_F(DisjoinL1ChainConfigsUnionBase, TestMergePolicy) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfig1, l1ChainConfig2;
  l1ChainConfig1.addOpL1MemSpec(memSpecOpA);
  l1ChainConfig2.addOpL1MemSpec(memSpecOpB);
  l1ChainConfig2.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfig1);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfig2);

  mlir::Operation *opD;
  opD = disjointL1ChainConfigsUnion.mergeL1ChainConfigs(opA, opB);

  ASSERT_EQ(opD, opB);
}

// Check if two ops belong to the same L1ChainConfig
TEST_F(DisjoinL1ChainConfigsUnionBase, TestOpsConnectivity) {
  mlir::Operation *opA, *opB, *opC;
  opA = createOp();
  opB = createOp();
  opC = createOp();

  OpL1MemSpec memSpecOpA, memSpecOpB, memSpecOpC;
  memSpecOpA.op = opA;
  memSpecOpB.op = opB;
  memSpecOpC.op = opC;

  L1ChainConfig l1ChainConfig1, l1ChainConfig2;
  l1ChainConfig1.addOpL1MemSpec(memSpecOpA);
  l1ChainConfig2.addOpL1MemSpec(memSpecOpB);
  l1ChainConfig2.addOpL1MemSpec(memSpecOpC);

  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfig1);
  disjointL1ChainConfigsUnion.insertL1ChainConfig(l1ChainConfig2);

  ASSERT_FALSE(disjointL1ChainConfigsUnion.connected(opA, opB));
  ASSERT_FALSE(disjointL1ChainConfigsUnion.connected(opA, opC));
  ASSERT_TRUE(disjointL1ChainConfigsUnion.connected(opB, opC));
}
