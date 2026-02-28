// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Scheduler/Scheduler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <random>

using namespace mlir::tt;

constexpr int NumberOfOps = 10;
constexpr int TensorDimX = 32;
constexpr int TensorDimY = 32;

// Define the operations that can be performed
enum class Operation : uint8_t { Add = 0, Sub, Div, Multiply, Max };

// Get a random operation
Operation getRandomOperation() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(
      0, static_cast<uint8_t>(Operation::Max) - 1);
  return static_cast<Operation>(dist(gen));
}

class SchedulerBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;

  void SetUp() override {
    // Initialize context and module
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<ttir::TTIRDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
    createFuncOp();
  }

  llvm::SmallVector<int64_t, 2> getTensorShape() {
    return {TensorDimX, TensorDimY};
  }

  mlir::Type getTensorType() {
    return mlir::RankedTensorType::get(getTensorShape(), builder.getF32Type());
  }

  mlir::Value createEmptyTensor() {
    return builder.create<mlir::tt::ttir::EmptyOp>(
        builder.getUnknownLoc(), getTensorShape(), builder.getF32Type());
  }

  mlir::func::FuncOp createFuncOp() {
    mlir::SmallVector<mlir::Type> input;
    input.push_back(getTensorType());

    mlir::SmallVector<mlir::Type> output;
    output.push_back(getTensorType());

    auto funcType = builder.getType<mlir::FunctionType>(
        mlir::TypeRange(input), mlir::TypeRange(output));
    func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "test",
                                              funcType);

    mlir::Block *block = func.addEntryBlock();
    block->addArgument(getTensorType(), builder.getUnknownLoc());
    block->addArgument(getTensorType(), builder.getUnknownLoc());

    builder.setInsertionPointToStart(block);

    return func;
  }

  void TearDown() override {}
};

// This tests chains all operations one after
// another, so output of scheduler order should
// be same as the order of operations created
TEST_F(SchedulerBase, FixedSchedule) {
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);

  // First operation has arg1 and arg2 (no DPS operand needed)
  ttir::TTIROp op = builder.create<ttir::AddOp>(builder.getUnknownLoc(),
                                                getTensorType(), lhs, rhs);

  // Create a chain of operations by using the result of the previous operation
  llvm::SmallVector<mlir::Value> operands = {rhs,
                                             op.getOperation()->getResult(0)};

  // Store the operations in a vector as we create them
  std::vector<ttir::TTIROp> ops;
  ops.push_back(op);

  for (std::size_t i = 1; i < NumberOfOps; i++) {
    mlir::Value lhs = operands[operands.size() - 2];
    mlir::Value rhs = operands[operands.size() - 1];
    op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), getTensorType(),
                                     lhs, rhs);
    operands.push_back(op.getOperation()->getResult(0));
    ops.push_back(op);
  }

  // Run scheduler to get the schedule
  mlir::tt::scheduler::Scheduler scheduler(&func);
  for (std::size_t i = 0; i < NumberOfOps; i++) {
    llvm::SmallVector<mlir::Operation *> schedulableOps =
        scheduler.getSchedulableOps();
    ASSERT_EQ(schedulableOps.size(), 1);
    ASSERT_TRUE(scheduler.hasUnscheduledOps());
    ;
    scheduler.scheduleOp(schedulableOps[0]);
  }

  ASSERT_FALSE(scheduler.hasUnscheduledOps());

  // Compare the schedule we got with the operations we created
  llvm::SmallVector<mlir::Operation *> schedule = scheduler.getSchedule();
  for (std::size_t i = 0; i < ops.size(); i++) {
    EXPECT_EQ(ops[i].getOperation(), schedule[i]);
  }

  // Just a sanity check that comparison is working
  EXPECT_NE(ops[0].getOperation(), schedule[1]);
}

// This tests the scheduler with a single operation
TEST_F(SchedulerBase, SingleOp) {
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);

  // First operation has arg1 and arg2 (no DPS operand needed)
  ttir::TTIROp op = builder.create<ttir::AddOp>(builder.getUnknownLoc(),
                                                getTensorType(), lhs, rhs);

  mlir::tt::scheduler::Scheduler scheduler(&func);
  ASSERT_TRUE(scheduler.hasUnscheduledOps());
  llvm::SmallVector<mlir::Operation *> schedulableOps =
      scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 1);
  scheduler.scheduleOp(schedulableOps[0]);
  ASSERT_FALSE(scheduler.hasUnscheduledOps());
  ASSERT_EQ(schedulableOps[0], op.getOperation());
}

// Test the scheduler with a fork in the graph
// First we have operation which works on arg1 and arg2
// Then we have two operations which work on the result of the first operation
// and arg1. Then we have forth operation which works on the result of the
// second and third operation. So the scheduler should first yield the first op
// and then the second and third op and after that the forth op.
TEST_F(SchedulerBase, VerifyFork) {
  // Create the first operation which works on arg1 and arg2
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
  ttir::TTIROp op = builder.create<ttir::AddOp>(builder.getUnknownLoc(),
                                                getTensorType(), lhs, rhs);

  std::vector<ttir::TTIROp> ops;
  ops.push_back(op);

  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = op.getOperation()->getResult(0);

  // Create the second operation which works on the result of the first
  // operation and arg1
  op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), getTensorType(),
                                   lhs, rhs);
  ops.push_back(op);
  op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), getTensorType(),
                                   lhs, rhs);
  ops.push_back(op);

  // Create the third operation which works on the result of the second and
  // third operation
  lhs = ops[ops.size() - 2].getOperation()->getResult(0);
  rhs = ops[ops.size() - 1].getOperation()->getResult(0);
  op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), getTensorType(),
                                   lhs, rhs);
  ops.push_back(op);

  mlir::tt::scheduler::Scheduler scheduler(&func);
  llvm::SmallVector<mlir::Operation *> schedulableOps =
      scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 1);

  scheduler.scheduleOp(schedulableOps[0]);
  schedulableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 2);

  scheduler.scheduleOp(schedulableOps[0]);
  schedulableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 1);

  scheduler.scheduleOp(schedulableOps[0]);
  schedulableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 1);

  scheduler.scheduleOp(schedulableOps[0]);
  ASSERT_FALSE(scheduler.hasUnscheduledOps());
}

// Test the scheduler with SplitQueryKeyValueAndSplitHeadsOp
// This op has 3 outputs (query, key, value), and each output has an operation
// after it. The scheduler should correctly initialize dependencies so that
// SplitQueryKeyValueAndSplitHeadsOp is scheduled first, then the 3 ops that
// use its outputs can be scheduled.
TEST_F(SchedulerBase, SplitQueryKeyValueAndSplitHeadsOp) {
  constexpr int batchSize = 1;
  constexpr int sequenceSize = 4;
  constexpr int numHeads = 2;
  constexpr int headDim = 8;
  constexpr int hiddenSize = numHeads * headDim;
  constexpr int inputHiddenSize = 3 * hiddenSize;

  llvm::SmallVector<int64_t> inputShape{batchSize, sequenceSize,
                                        inputHiddenSize};
  llvm::SmallVector<int64_t> outputShape{batchSize, numHeads, sequenceSize,
                                         headDim};

  mlir::Value inputTensor = builder.create<ttir::EmptyOp>(
      builder.getUnknownLoc(), inputShape, builder.getF32Type());

  mlir::Type queryType =
      mlir::RankedTensorType::get(outputShape, builder.getF32Type());
  mlir::Type keyType =
      mlir::RankedTensorType::get(outputShape, builder.getF32Type());
  mlir::Type valueType =
      mlir::RankedTensorType::get(outputShape, builder.getF32Type());

  auto splitOp = builder.create<ttir::SplitQueryKeyValueAndSplitHeadsOp>(
      builder.getUnknownLoc(), queryType, keyType, valueType, inputTensor,
      /*kv_input_tensor=*/nullptr, builder.getUI32IntegerAttr(numHeads),
      /*num_kv_heads=*/nullptr, builder.getBoolAttr(false));

  auto outputType =
      mlir::RankedTensorType::get(getTensorShape(), builder.getF32Type());
  mlir::Value arg0 = func.getBody().getBlocks().front().getArgument(0);
  auto queryConsumerOp = builder.create<ttir::AddOp>(
      builder.getUnknownLoc(), outputType, splitOp.getQuery(), arg0);
  auto keyConsumerOp = builder.create<ttir::AddOp>(
      builder.getUnknownLoc(), outputType, splitOp.getKey(), arg0);
  auto valueConsumerOp = builder.create<ttir::AddOp>(
      builder.getUnknownLoc(), outputType, splitOp.getValue(), arg0);

  mlir::tt::scheduler::Scheduler scheduler(&func);
  llvm::SmallVector<mlir::Operation *> schedulableOps =
      scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 1);
  ASSERT_EQ(schedulableOps[0], splitOp.getOperation());

  scheduler.scheduleOp(schedulableOps[0]);
  schedulableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(schedulableOps.size(), 3);

  ASSERT_EQ(schedulableOps[0], queryConsumerOp.getOperation());
  ASSERT_EQ(schedulableOps[1], keyConsumerOp.getOperation());
  ASSERT_EQ(schedulableOps[2], valueConsumerOp.getOperation());
}
