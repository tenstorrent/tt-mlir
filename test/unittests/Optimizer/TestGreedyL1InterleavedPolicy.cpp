// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/Analysis/GreedyL1InterleavedPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

using namespace mlir::tt::ttnn;

constexpr int TensorDimX = 128;
constexpr int TensorDimY = 128;

class GreedyL1InterleavedPolicyBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);
  mlir::func::FuncOp func;
  mlir::tt::ttcore::DeviceAttr deviceAttr;

  using OpMemSpec = GreedyL1InterleavedPolicy::OpMemSpec;
  using GreedyPolicyChoice = GreedyL1InterleavedPolicy::GreedyPolicyChoice;
  using L1Usage = GreedyL1InterleavedPolicy::L1Usage;

  void SetUp() override {
    context.loadDialect<TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
    createFuncOp();
    deviceAttr = mlir::tt::ttcore::lookupDevice(func);
  }

  llvm::SmallVector<int64_t, 2> getTensorShape() {
    return {TensorDimX, TensorDimY};
  }

  mlir::RankedTensorType getTensorRankedType() {
    return mlir::RankedTensorType::get(getTensorShape(), builder.getF32Type());
  }

  mlir::Value createEmptyTensor() {
    ShapeAttr shapeAttr = ShapeAttr::get(&context, getTensorShape());
    return builder.create<OnesOp>(builder.getUnknownLoc(),
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

  void addConfigForOp(
      mlir::Operation *op,
      llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> &legalConfigs,
      BufferType memorySpace, TensorMemoryLayout tensorMemoryLayout) {
    TensorMemoryLayoutAttr tensorMemoryLayoutAttr =
        TensorMemoryLayoutAttr::get(&context, tensorMemoryLayout);
    mlir::tt::ttcore::GridAttr gridAttr =
        mlir::tt::ttcore::GridAttr::get(&context);
    if (isL1BufferType(memorySpace)) {
      gridAttr = mlir::tt::ttcore::GridAttr::get(&context, {8, 8});
    }

    if (legalConfigs.find(op) == legalConfigs.end()) {
      legalConfigs[op] = std::vector<OpConfig>{TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(),
          mlir::tt::ttcore::TileType::get(builder.getF32Type()), memorySpace,
          gridAttr, tensorMemoryLayoutAttr)};
    } else {
      legalConfigs[op].push_back(TTNNLayoutAttr::get(
          &context, getTensorRankedType().getShape(),
          mlir::tt::ttcore::TileType::get(builder.getF32Type()), memorySpace,
          gridAttr, tensorMemoryLayoutAttr));
    }
  }

  void prepareOpForGreedyConfigPicker(
      mlir::Operation *op, uint64_t outputL1Usage, uint64_t requiredL1Usage,
      llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> &legalConfigs,
      llvm::DenseMap<mlir::Operation *, L1Usage> &opsL1Usage) {

    // Add two legal configs for the op with different buffer
    // types: DRAM and L1.
    addConfigForOp(op, legalConfigs, BufferType::DRAM,
                   TensorMemoryLayout::Interleaved);
    addConfigForOp(op, legalConfigs, BufferType::L1,
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
  llvm::DenseMap<mlir::Operation *, std::vector<OpConfig>> legalConfigs;
  llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::Operation *>>
      schedule;
  llvm::DenseMap<mlir::Operation *, L1Usage> opsL1Usage;
  constexpr uint64_t usableL1CacheSize = 15;

  // Create operand A
  mlir::Value lhs = func.getBody().getBlocks().front().getArgument(0);
  mlir::Value rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opA =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  uint64_t outputL1Usage = 2;
  uint64_t requiredL1Usage = 8;
  prepareOpForGreedyConfigPicker(opA, outputL1Usage, requiredL1Usage,
                                 legalConfigs, opsL1Usage);

  // Create operand B
  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opB =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  outputL1Usage = 3;
  requiredL1Usage = 7;
  prepareOpForGreedyConfigPicker(opB, outputL1Usage, requiredL1Usage,
                                 legalConfigs, opsL1Usage);

  // Create operand C
  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opC =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  outputL1Usage = 1;
  requiredL1Usage = 9;
  prepareOpForGreedyConfigPicker(opC, outputL1Usage, requiredL1Usage,
                                 legalConfigs, opsL1Usage);

  // Create base op D
  lhs = func.getBody().getBlocks().front().getArgument(0);
  rhs = func.getBody().getBlocks().front().getArgument(1);
  mlir::Operation *opD =
      builder.create<AddOp>(builder.getUnknownLoc(), lhs.getType(), lhs, rhs);
  outputL1Usage = 4;
  requiredL1Usage = 0;
  prepareOpForGreedyConfigPicker(opD, outputL1Usage, requiredL1Usage,
                                 legalConfigs, opsL1Usage);

  // Run greedy config picker policy
  GreedyL1InterleavedPolicy l1InterleavedPolicy(
      nullptr, l1ChainConfigs, legalConfigs, schedule, usableL1CacheSize);
  GreedyPolicyChoice greedyConfig =
      l1InterleavedPolicy.getGreedyConfig(opD, opsL1Usage);

  // Sanity checks
  ASSERT_TRUE(greedyConfig.baseOp == opD);
  ASSERT_TRUE(greedyConfig.configs.size() == 4);
  ASSERT_TRUE(greedyConfig.precedence.size() == 3);

  // All layouts should be using L1 buffer type
  for (const auto &[op, config] : greedyConfig.configs) {
    ASSERT_TRUE(config.outputLayout.hasL1BufferType());
  }

  // Precedence order for op D should be: C, A, B
  ASSERT_EQ(greedyConfig.precedence[0], opC);
  ASSERT_EQ(greedyConfig.precedence[1], opA);
  ASSERT_EQ(greedyConfig.precedence[2], opB);
}
