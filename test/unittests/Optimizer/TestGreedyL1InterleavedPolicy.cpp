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

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "ttmlir/Dialect/TTNN/Analysis/GreedyL1InterleavedPolicy.h"

using namespace mlir::tt::ttnn;

constexpr int TensorDimX = 128;
constexpr int TensorDimY = 128;

class GreedyL1InterleavedPolicyBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;
  mlir::tt::DeviceAttr deviceAttr;

  using OpMemSpec = GreedyL1InterleavedPolicy::OpMemSpec;
  using OpConfig = GreedyL1InterleavedPolicy::OpConfig;
  using L1Usage = GreedyL1InterleavedPolicy::L1Usage;

  void SetUp() override {
    context.loadDialect<TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    createFuncOp();
    deviceAttr = mlir::tt::getCurrentScopeDevice(func);
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

  void addLayoutForOp(mlir::Operation *op,
                      llvm::DenseMap<mlir::Operation *,
                                     std::vector<TTNNLayoutAttr>> &legalLayouts,
                      BufferType memorySpace,
                      TensorMemoryLayout tensorMemoryLayout) {
    TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
        TensorMemoryLayoutAttr::get(&context, tensorMemoryLayout);
    if (legalLayouts.find(op) == legalLayouts.end()) {
      legalLayouts[op] = std::vector<TTNNLayoutAttr>{TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(),
          mlir::tt::TileType::get(&context, builder.getF32Type()), memorySpace,
          mlir::tt::GridAttr::get(&context, {8, 8}), tensorMemoryLayoutAttr)};
    } else {
      legalLayouts[op].push_back(TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(),
          mlir::tt::TileType::get(&context, builder.getF32Type()), memorySpace,
          mlir::tt::GridAttr::get(&context, {8, 8}), tensorMemoryLayoutAttr));
    }
  }

  void prepareOpForGreedyConfigPicker(
      mlir::Operation *op, uint64_t outputL1Usage, uint64_t requiredL1Usage,
      llvm::DenseMap<mlir::Operation *, std::vector<TTNNLayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<mlir::Operation *, L1Usage> &opsL1Usage) {

    // Add two legal layouts for the op with different buffer
    // types: DRAM and L1.
    addLayoutForOp(op, legalLayouts, BufferType::DRAM,
                   TensorMemoryLayout::Interleaved);
    addLayoutForOp(op, legalLayouts, BufferType::L1,
                   TensorMemoryLayout::Interleaved);

    L1Usage l1Usage;
    l1Usage.outputL1Usage = outputL1Usage;
    l1Usage.requiredL1Usage = requiredL1Usage;
    opsL1Usage[op] = l1Usage;
  }

  void TearDown() override {}
};

TEST_F(GreedyL1InterleavedPolicyBase, VerifyGreedyPolicy) {
  std::vector<L1ChainConfig> l1ChainConfigs;
  llvm::DenseMap<mlir::Operation *, std::vector<TTNNLayoutAttr>> legalLayouts;
  llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>>
      schedule;
  llvm::DenseMap<mlir::Operation *, L1Usage> opsL1Usage;
  constexpr uint64_t usableL1CacheSize = 15;

  // Create operand A
  mlir::Value dest = createEmptyTensor();
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opA =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  uint64_t outputL1Usage = 2;
  uint64_t requiredL1Usage = 8;
  prepareOpForGreedyConfigPicker(opA, outputL1Usage, requiredL1Usage,
                                 legalLayouts, opsL1Usage);

  // Create operand B
  dest = createEmptyTensor();
  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opB =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  outputL1Usage = 3;
  requiredL1Usage = 7;
  prepareOpForGreedyConfigPicker(opB, outputL1Usage, requiredL1Usage,
                                 legalLayouts, opsL1Usage);

  // Create operand C
  dest = createEmptyTensor();
  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opC =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  outputL1Usage = 1;
  requiredL1Usage = 9;
  prepareOpForGreedyConfigPicker(opC, outputL1Usage, requiredL1Usage,
                                 legalLayouts, opsL1Usage);

  // Create base op D
  dest = createEmptyTensor();
  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opD =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs, dest);
  outputL1Usage = 4;
  requiredL1Usage = 0;
  prepareOpForGreedyConfigPicker(opD, outputL1Usage, requiredL1Usage,
                                 legalLayouts, opsL1Usage);

  // Run greedy config picker policy
  GreedyL1InterleavedPolicy l1InterleavedPolicy(
      nullptr, l1ChainConfigs, legalLayouts, schedule, usableL1CacheSize);
  OpConfig greedyConfig = l1InterleavedPolicy.getGreedyConfig(opD, opsL1Usage);

  // Sanity checks
  ASSERT_TRUE(greedyConfig.baseOp == opD);
  ASSERT_TRUE(greedyConfig.layouts.size() == 4);
  ASSERT_TRUE(greedyConfig.precedence.size() == 3);

  // All layouts should be using L1 buffer type
  for (const auto &[op, layout] : greedyConfig.layouts) {
    ASSERT_TRUE(layout.hasL1BufferType());
  }

  // Precedence order for op D should be: C, A, B
  ASSERT_EQ(greedyConfig.precedence[0], opC);
  ASSERT_EQ(greedyConfig.precedence[1], opA);
  ASSERT_EQ(greedyConfig.precedence[2], opB);
}
