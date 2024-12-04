// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <mlir/IR/Operation.h>

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

TEST_F(DisjoinL1ChainConfigsUnionBase, TestInsertOp) {
  mlir::Operation *opA = createOp();
  mlir::Operation *opB = createOp();
  mlir::Operation *opC = createOp();

  disjointL1ChainConfigsUnion.insertOp(opA);
  disjointL1ChainConfigsUnion.insertOp(opB);
  disjointL1ChainConfigsUnion.insertOp(opC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.size(), 3);
}

TEST_F(DisjoinL1ChainConfigsUnionBase, TestFindRepresentativeOpSimple) {
  mlir::Operation *opA = createOp();
  mlir::Operation *opB = createOp();
  mlir::Operation *opC = createOp();

  disjointL1ChainConfigsUnion.insertOp(opA);
  disjointL1ChainConfigsUnion.insertOp(opB);
  disjointL1ChainConfigsUnion.insertOp(opC);

  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opA), opA);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opB), opB);
  ASSERT_EQ(disjointL1ChainConfigsUnion.findRepresentativeOp(opC), opC);
}
