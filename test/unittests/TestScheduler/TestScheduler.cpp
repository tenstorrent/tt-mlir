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
  mlir::Value dest = createEmptyTensor();
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);

  // First operation has arg1 and arg2 and %0 as dps operand
  ttir::TTIROp op =
      builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);

  // Create a chain of operations by using the result of the previous operation
  llvm::SmallVector<mlir::Value> operands = {rhs,
                                             op.getOperation()->getResult(0)};

  // Store the operations in a vector as we create them
  std::vector<ttir::TTIROp> ops;
  ops.push_back(op);

  for (std::size_t i = 1; i < NumberOfOps; i++) {
    mlir::Value lhs = operands[operands.size() - 2];
    mlir::Value rhs = operands[operands.size() - 1];
    dest = createEmptyTensor();
    op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
    operands.push_back(op.getOperation()->getResult(0));
    ops.push_back(op);
  }

  // Run scheduler to get the schedule
  mlir::tt::scheduler::Scheduler scheduler(&func);
  for (std::size_t i = 0; i < NumberOfOps; i++) {
    llvm::SmallVector<mlir::Operation *> scheduleableOps =
        scheduler.getSchedulableOps();
    ASSERT_EQ(scheduleableOps.size(), 1);
    ASSERT_TRUE(scheduler.hasUnscheduledOps());
    ;
    scheduler.scheduleOp(scheduleableOps[0]);
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
  mlir::Value dest = createEmptyTensor();
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);

  // First operation has arg1 and arg2 and %0 as dps operand
  ttir::TTIROp op =
      builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);

  mlir::tt::scheduler::Scheduler scheduler(&func);
  ASSERT_TRUE(scheduler.hasUnscheduledOps());
  llvm::SmallVector<mlir::Operation *> scheduleableOps =
      scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 1);
  scheduler.scheduleOp(scheduleableOps[0]);
  ASSERT_FALSE(scheduler.hasUnscheduledOps());
  ASSERT_EQ(scheduleableOps[0], op.getOperation());
}

// Test the scheduler with a fork in the graph
// First we have operation which works on arg1 and arg2
// Then we have two operations which work on the result of the first operation
// and arg1. Then we have forth operation which works on the result of the
// second and third operation. So the scheduler should first yield the first op
// and then the second and third op and after that the forth op.
TEST_F(SchedulerBase, VerifyFork) {
  // Create the first operation which works on arg1 and arg2
  mlir::Value dest = createEmptyTensor();
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
  ttir::TTIROp op =
      builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);

  std::vector<ttir::TTIROp> ops;
  ops.push_back(op);

  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = op.getOperation()->getResult(0);

  // Create the second operation which works on the result of the first
  // operation and arg1
  dest = createEmptyTensor();
  op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  ops.push_back(op);
  dest = createEmptyTensor();
  op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  ops.push_back(op);

  // Create the third operation which works on the result of the second and
  // third operation
  lhs = ops[ops.size() - 2].getOperation()->getResult(0);
  rhs = ops[ops.size() - 1].getOperation()->getResult(0);
  dest = createEmptyTensor();
  op = builder.create<ttir::AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  ops.push_back(op);

  mlir::tt::scheduler::Scheduler scheduler(&func);
  llvm::SmallVector<mlir::Operation *> scheduleableOps =
      scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 1);

  scheduler.scheduleOp(scheduleableOps[0]);
  scheduleableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 2);

  scheduler.scheduleOp(scheduleableOps[0]);
  scheduleableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 1);

  scheduler.scheduleOp(scheduleableOps[0]);
  scheduleableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 1);

  scheduler.scheduleOp(scheduleableOps[0]);
  ASSERT_FALSE(scheduler.hasUnscheduledOps());
}

// Test the scheduler with SplitQueryKeyValueAndSplitHeadsOp
// This op has 3 outputs (query, key, value), and each output has an operation
// after it. The scheduler should correctly initialize dependencies so that
// SplitQueryKeyValueAndSplitHeadsOp is scheduled first, then the 3 ops that
// use its outputs can be scheduled.
TEST_F(SchedulerBase, SplitQueryKeyValueAndSplitHeadsOp) {
  // Set up tensor shapes for SplitQueryKeyValueAndSplitHeadsOp
  // Input: [batch_size, sequence_size, 3 * hidden_size]
  // Outputs: [batch_size, num_heads, sequence_size, head_dim]
  constexpr int batchSize = 1;
  constexpr int sequenceSize = 4;
  constexpr int numHeads = 2;
  constexpr int headDim = 8;
  constexpr int hiddenSize = numHeads * headDim;  // 16
  constexpr int inputHiddenSize = 3 * hiddenSize; // 48

  llvm::SmallVector<int64_t> inputShape{batchSize, sequenceSize,
                                        inputHiddenSize};
  llvm::SmallVector<int64_t> outputShape{batchSize, numHeads, sequenceSize,
                                         headDim};

  // Create input tensor
  mlir::Value inputTensor = builder.create<ttir::EmptyOp>(
      builder.getUnknownLoc(), inputShape, builder.getF32Type());

  // Create output tensors for DPS operands
  mlir::Value queryOutput = builder.create<ttir::EmptyOp>(
      builder.getUnknownLoc(), outputShape, builder.getF32Type());
  mlir::Value keyOutput = builder.create<ttir::EmptyOp>(
      builder.getUnknownLoc(), outputShape, builder.getF32Type());
  mlir::Value valueOutput = builder.create<ttir::EmptyOp>(
      builder.getUnknownLoc(), outputShape, builder.getF32Type());

  // Create attributes
  mlir::IntegerAttr numHeadsAttr = builder.getUI32IntegerAttr(numHeads);
  mlir::BoolAttr transposeKeyAttr = builder.getBoolAttr(false);

  // Create result types for the 3 outputs
  mlir::Type queryType =
      mlir::RankedTensorType::get(outputShape, builder.getF32Type());
  mlir::Type keyType =
      mlir::RankedTensorType::get(outputShape, builder.getF32Type());
  mlir::Type valueType =
      mlir::RankedTensorType::get(outputShape, builder.getF32Type());

  // Create SplitQueryKeyValueAndSplitHeadsOp
  auto splitOp = builder.create<ttir::SplitQueryKeyValueAndSplitHeadsOp>(
      builder.getUnknownLoc(), queryType, keyType, valueType, inputTensor,
      /*kv_input_tensor=*/nullptr, queryOutput, keyOutput, valueOutput,
      numHeadsAttr, /*num_kv_heads=*/nullptr, transposeKeyAttr);

  // Get the 3 outputs
  mlir::Value query = splitOp.getQuery();
  mlir::Value key = splitOp.getKey();
  mlir::Value value = splitOp.getValue();

  // Create operations that use each of the 3 outputs
  mlir::Value arg0 = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value destQuery = createEmptyTensor();
  ttir::TTIROp queryOp = builder.create<ttir::AddOp>(builder.getUnknownLoc(),
                                                     query, arg0, destQuery);

  mlir::Value destKey = createEmptyTensor();
  ttir::TTIROp keyOp =
      builder.create<ttir::AddOp>(builder.getUnknownLoc(), key, arg0, destKey);

  mlir::Value destValue = createEmptyTensor();
  ttir::TTIROp valueOp = builder.create<ttir::AddOp>(builder.getUnknownLoc(),
                                                     value, arg0, destValue);

  // Initialize scheduler
  mlir::tt::scheduler::Scheduler scheduler(&func);

  // Verify scheduler is initialized correctly
  // SplitQueryKeyValueAndSplitHeadsOp should be schedulable first (no
  // dependencies)
  ASSERT_TRUE(scheduler.hasUnscheduledOps());
  llvm::SmallVector<mlir::Operation *> scheduleableOps =
      scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 1);
  ASSERT_EQ(scheduleableOps[0], splitOp.getOperation());

  // Schedule SplitQueryKeyValueAndSplitHeadsOp
  scheduler.scheduleOp(scheduleableOps[0]);

  // Now all 3 ops that use the outputs should be schedulable
  scheduleableOps = scheduler.getSchedulableOps();
  ASSERT_EQ(scheduleableOps.size(), 3);
  ASSERT_TRUE(scheduler.hasUnscheduledOps());

  // Verify all 3 ops are in the schedulable list
  bool foundQueryOp = false;
  bool foundKeyOp = false;
  bool foundValueOp = false;
  for (mlir::Operation *op : scheduleableOps) {
    if (op == queryOp.getOperation()) {
      foundQueryOp = true;
    } else if (op == keyOp.getOperation()) {
      foundKeyOp = true;
    } else if (op == valueOp.getOperation()) {
      foundValueOp = true;
    }
  }
  ASSERT_TRUE(foundQueryOp);
  ASSERT_TRUE(foundKeyOp);
  ASSERT_TRUE(foundValueOp);

  // Schedule all 3 ops
  for (mlir::Operation *op : scheduleableOps) {
    scheduler.scheduleOp(op);
  }

  // Verify all ops are scheduled
  ASSERT_FALSE(scheduler.hasUnscheduledOps());
  llvm::SmallVector<mlir::Operation *> schedule = scheduler.getSchedule();
  ASSERT_EQ(schedule.size(), 4); // 1 split op + 3 add ops
}
